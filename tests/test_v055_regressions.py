"""
Regression tests for v0.5.5.

1. check_with_evidence(): evidence that has no len() (generator, iterator,
   map/filter object) is converted to a list once, before the empty check.
   Previously a generator was always truthy, so an empty one skipped
   NO_EVIDENCE, and a one-shot iterator could be consumed by the first
   read inside compare_fn. Anything with len() (list, tuple, numpy array)
   is passed through as the same object.
2. check()/acheck(): a non-callable `validator` now raises TypeError
   before any call_fn runs. Previously it burned max_retries + 1 LLM calls
   and returned UNCERTAIN.
3. __version__ and pyproject.toml must agree.
"""

import asyncio
import re
from pathlib import Path

import pytest

import booth

GOOD = '{"answer": "Paris", "confidence": 0.95}'
PROMPT = "What is the capital of France?"


def _call_fn_counter():
    state = {"count": 0}

    def call_fn(prompt):
        state["count"] += 1
        return GOOD

    call_fn.state = state
    return call_fn


def _async_call_fn_counter():
    state = {"count": 0}

    async def call_fn(prompt):
        state["count"] += 1
        return GOOD

    call_fn.state = state
    return call_fn


def _capture():
    """compare_fn that records what it received and returns True."""
    seen = {"calls": 0, "evidence": None}

    def compare_fn(answer, evidence):
        seen["calls"] += 1
        seen["evidence"] = evidence
        return True

    compare_fn.seen = seen
    return compare_fn


# ---------------------------------------------------------------------
# Fix 1: evidence without len()
# ---------------------------------------------------------------------

def _make_generator():
    yield "a"
    yield "b"


@pytest.mark.parametrize(
    "make_evidence",
    [
        lambda: _make_generator(),
        lambda: (x for x in ["a", "b"]),
        lambda: iter(["a", "b"]),
        lambda: map(str, ["a", "b"]),
        lambda: filter(None, ["a", "b"]),
    ],
    ids=["generator", "genexp", "iterator", "map", "filter"],
)
def test_one_shot_evidence_reaches_compare_fn_as_a_full_list(make_evidence):
    compare = _capture()
    result = booth.check_with_evidence("an answer", make_evidence(), compare)

    assert result.status == booth.ACCEPTED
    assert compare.seen["calls"] == 1
    assert isinstance(compare.seen["evidence"], list)
    assert compare.seen["evidence"] == ["a", "b"]


def test_comparator_that_reads_evidence_twice_sees_it_both_times():
    """The one-shot iterator bug: the first read used to exhaust the
    generator, so a second read saw nothing."""
    def compare_fn(answer, evidence):
        first = list(evidence)
        second = list(evidence)
        return bool(first) and first == second

    result = booth.check_with_evidence("an answer", _make_generator(), compare_fn)
    assert result.status == booth.ACCEPTED


@pytest.mark.parametrize(
    "make_empty",
    [
        lambda: (x for x in []),
        lambda: iter([]),
        lambda: map(str, []),
    ],
    ids=["genexp", "iterator", "map"],
)
def test_empty_one_shot_evidence_is_no_evidence(make_empty):
    compare = _capture()
    result = booth.check_with_evidence("an answer", make_empty(), compare)

    assert result.status == booth.UNCERTAIN
    assert result.reason == booth.NO_EVIDENCE
    assert compare.seen["calls"] == 0


def test_list_evidence_is_passed_through_as_the_same_object():
    compare = _capture()
    evidence = ["a", "b"]
    booth.check_with_evidence("an answer", evidence, compare)

    assert compare.seen["evidence"] is evidence


def test_tuple_evidence_is_passed_through_as_the_same_object():
    compare = _capture()
    evidence = ("a", "b")
    booth.check_with_evidence("an answer", evidence, compare)

    assert compare.seen["evidence"] is evidence


def test_empty_list_still_no_evidence():
    compare = _capture()
    result = booth.check_with_evidence("an answer", [], compare)

    assert result.reason == booth.NO_EVIDENCE
    assert compare.seen["calls"] == 0


def test_none_evidence_still_no_evidence_and_does_not_crash():
    compare = _capture()
    result = booth.check_with_evidence("an answer", None, compare)

    assert result.status == booth.UNCERTAIN
    assert result.reason == booth.NO_EVIDENCE
    assert compare.seen["calls"] == 0


def test_numpy_array_evidence_is_passed_through_untouched():
    np = pytest.importorskip("numpy")
    compare = _capture()
    evidence = np.array(["a", "b"])
    result = booth.check_with_evidence("an answer", evidence, compare)

    assert result.status == booth.ACCEPTED
    assert compare.seen["evidence"] is evidence


def test_empty_numpy_array_is_no_evidence():
    np = pytest.importorskip("numpy")
    compare = _capture()
    result = booth.check_with_evidence("an answer", np.array([]), compare)

    assert result.reason == booth.NO_EVIDENCE
    assert compare.seen["calls"] == 0


def test_empty_answer_returns_before_evidence_is_consumed():
    """The answer check still runs first, so a rejected answer does not
    drain the caller's iterator."""
    it = iter(["a", "b"])
    result = booth.check_with_evidence("", it, lambda a, e: True)

    assert result.reason == booth.EMPTY_ANSWER
    assert list(it) == ["a", "b"]


def test_generator_that_raises_propagates_to_the_caller():
    """Decision for v0.5.5: an error inside the caller's own generator is
    not a compare_fn failure, so it propagates instead of becoming
    COMPARE_FAILED."""
    def bad_gen():
        yield "a"
        raise RuntimeError("retrieval blew up")

    compare = _capture()
    with pytest.raises(RuntimeError, match="retrieval blew up"):
        booth.check_with_evidence("an answer", bad_gen(), compare)

    assert compare.seen["calls"] == 0


def test_generator_raising_typeerror_is_not_swallowed():
    """Guards the subtle case: a TypeError raised inside the generator
    must not be mistaken for 'evidence is not iterable'."""
    def bad_gen():
        yield "a"
        raise TypeError("inner type error")

    compare = _capture()
    with pytest.raises(TypeError, match="inner type error"):
        booth.check_with_evidence("an answer", bad_gen(), compare)

    assert compare.seen["calls"] == 0


def test_generator_evidence_works_with_score_and_threshold():
    result = booth.check_with_evidence(
        "an answer",
        _make_generator(),
        lambda a, e: 0.4 if len(e) == 2 else 1.0,
        evidence_threshold=0.5,
    )

    assert result.status == booth.BLOCKED
    assert result.reason == booth.EVIDENCE_DISAGREES
    assert result.evidence_agreement == 0.4


# ---------------------------------------------------------------------
# Fix 2: non-callable validator rejected upfront
# ---------------------------------------------------------------------

@pytest.mark.parametrize("bad_validator", ["not_callable", 123, [], {}, 3.14])
def test_check_rejects_non_callable_validator(bad_validator):
    call_fn = _call_fn_counter()

    with pytest.raises(TypeError, match="validator must be callable"):
        booth.check(call_fn, PROMPT, validator=bad_validator)

    assert call_fn.state["count"] == 0


def test_check_non_callable_validator_rejected_before_any_retry():
    call_fn = _call_fn_counter()

    with pytest.raises(TypeError, match="validator must be callable"):
        booth.check(call_fn, PROMPT, validator="nope", max_retries=3)

    assert call_fn.state["count"] == 0


@pytest.mark.parametrize("bad_validator", ["not_callable", 123, [], {}, 3.14])
def test_acheck_rejects_non_callable_validator(bad_validator):
    call_fn = _async_call_fn_counter()

    async def run():
        with pytest.raises(TypeError, match="validator must be callable"):
            await booth.acheck(call_fn, PROMPT, validator=bad_validator)

    asyncio.run(run())
    assert call_fn.state["count"] == 0


def test_acheck_non_callable_validator_rejected_before_any_retry():
    call_fn = _async_call_fn_counter()

    async def run():
        with pytest.raises(TypeError, match="validator must be callable"):
            await booth.acheck(call_fn, PROMPT, validator="nope", max_retries=3)

    asyncio.run(run())
    assert call_fn.state["count"] == 0


def test_check_none_validator_still_allowed():
    result = booth.check(_call_fn_counter(), PROMPT, validator=None)
    assert result.ok


def test_acheck_none_validator_still_allowed():
    result = asyncio.run(booth.acheck(_async_call_fn_counter(), PROMPT, validator=None))
    assert result.ok


def test_check_valid_validator_still_works():
    result = booth.check(_call_fn_counter(), PROMPT, validator=lambda answer: answer == "Paris")
    assert result.status == booth.ACCEPTED


def test_check_failing_validator_still_reaches_uncertain():
    """A callable validator that rejects the answer keeps its old behavior."""
    result = booth.check(
        _call_fn_counter(),
        PROMPT,
        max_retries=1,
        validator=lambda answer: (False, "nope"),
    )

    assert result.status == booth.UNCERTAIN
    assert result.method == "validation"
    assert result.attempts[-1].validation_error == "nope"


def test_check_callable_object_validator_is_accepted():
    class Validator:
        def __call__(self, answer):
            return True

    result = booth.check(_call_fn_counter(), PROMPT, validator=Validator())
    assert result.ok


def test_acheck_valid_validator_still_works():
    result = asyncio.run(
        booth.acheck(_async_call_fn_counter(), PROMPT, validator=lambda answer: True)
    )
    assert result.status == booth.ACCEPTED


def test_async_validator_is_callable_so_it_gets_the_coroutine_message_not_typeerror():
    """An async def validator passes the new callable() check and is
    still handled by the existing 'must be synchronous' rejection."""
    async def async_validator(answer):
        return True

    result = booth.check(_call_fn_counter(), PROMPT, max_retries=0, validator=async_validator)

    assert result.status == booth.UNCERTAIN
    assert result.attempts[-1].passed_validation is False
    assert "coroutine" in result.attempts[-1].validation_error


# ---------------------------------------------------------------------
# Housekeeping: version consistency
# ---------------------------------------------------------------------

def test_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("pyproject.toml not found (running outside the repo checkout)")

    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), re.M)
    assert match is not None, "no version line found in pyproject.toml"
    assert booth.__version__ == match.group(1)