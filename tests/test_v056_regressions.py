"""
Regression tests for v0.5.6.

1. check_with_evidence(): `evidence` is annotated Iterable[str], matching the
   v0.5.5 behavior of accepting generators and iterators. Before, the hint
   said Sequence[str], so type checkers rejected what the code accepted.
2. Evidence that has a length but is not a Sequence (a set) still works.
3. Leftover "# NEW" development markers are gone from core.py.
4. The README, version and package metadata stay in step (version check
   lives in test_v055_regressions.py and keeps running).
"""

import inspect
import typing

import pytest

import booth
import booth.core as core


def _capture():
    seen = {"calls": 0, "evidence": None}

    def compare_fn(answer, evidence):
        seen["calls"] += 1
        seen["evidence"] = evidence
        return True

    compare_fn.seen = seen
    return compare_fn


# ---------------------------------------------------------------------
# Fix 1: the type hint matches the behavior
# ---------------------------------------------------------------------

def test_evidence_is_annotated_as_iterable_of_str():
    hints = typing.get_type_hints(booth.check_with_evidence)
    assert hints["evidence"] == typing.Iterable[str]


def test_compare_fn_type_still_receives_a_sequence():
    """CompareFn stays Sequence[str]: iterators are converted to a list
    before compare_fn is called."""
    args = typing.get_args(booth.CompareFn)
    # args[0] is the parameter list, args[1] the return type
    assert typing.Sequence[str] in args[0]


def test_materialize_evidence_is_annotated_to_return_a_sequence():
    hints = typing.get_type_hints(core._materialize_evidence)
    assert hints["return"] == typing.Sequence[str]


# ---------------------------------------------------------------------
# Behavior unchanged by the refactor to a separate variable
# ---------------------------------------------------------------------

def test_generator_still_reaches_compare_fn_as_a_list():
    compare = _capture()
    result = booth.check_with_evidence("an answer", (e for e in ["a", "b"]), compare)

    assert result.status == booth.ACCEPTED
    assert compare.seen["evidence"] == ["a", "b"]
    assert isinstance(compare.seen["evidence"], list)


def test_list_is_still_passed_through_as_the_same_object():
    compare = _capture()
    evidence = ["a", "b"]
    booth.check_with_evidence("an answer", evidence, compare)

    assert compare.seen["evidence"] is evidence


def test_empty_generator_is_still_no_evidence():
    compare = _capture()
    result = booth.check_with_evidence("an answer", (e for e in []), compare)

    assert result.reason == booth.NO_EVIDENCE
    assert compare.seen["calls"] == 0


def test_none_evidence_is_still_no_evidence():
    compare = _capture()
    result = booth.check_with_evidence("an answer", None, compare)

    assert result.reason == booth.NO_EVIDENCE
    assert compare.seen["calls"] == 0


# ---------------------------------------------------------------------
# Fix 2: a set is Iterable[str] and has len(), so it passes straight through
# ---------------------------------------------------------------------

def test_set_evidence_is_accepted_and_passed_through_as_the_same_object():
    compare = _capture()
    evidence = {"a", "b"}
    result = booth.check_with_evidence("an answer", evidence, compare)

    assert result.status == booth.ACCEPTED
    assert compare.seen["evidence"] is evidence


def test_empty_set_is_no_evidence():
    compare = _capture()
    result = booth.check_with_evidence("an answer", set(), compare)

    assert result.reason == booth.NO_EVIDENCE
    assert compare.seen["calls"] == 0


# ---------------------------------------------------------------------
# Cleanup: no development markers left in the source
# ---------------------------------------------------------------------

def test_no_new_markers_left_in_core_source():
    source = inspect.getsource(core)
    assert "# NEW" not in source