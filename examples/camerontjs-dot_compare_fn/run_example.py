"""
Compare a weak lexical BOOTH evidence checker against a pinned DeBERTa NLI checker.

This is an illustration of compare_fn behavior, not a factual-verification benchmark.
No LLM generator, retrieval system, API key, or BOOTH code changes are required.
Both comparators receive exactly the same authored answer and evidence per case.

From the BOOTH checkout root on macOS/Linux, in a clean Python 3.9+
environment (tested on Python 3.9.6, macOS arm64):

    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install boothpy==0.5.4 torch==2.8.0 transformers==4.57.6 sentencepiece==0.2.2
    python examples/camerontjs-dot_compare_fn/run_example.py

This example imports boothpy==0.5.4 installed from PyPI, NOT BOOTH from the
checkout. Do not install the checkout in editable mode or put its src/ on
PYTHONPATH in the example environment.

The first execution downloads about 380 MB of pinned model/tokenizer files if
not cached (observed: 369 MB weights plus 11 MB tokenizer/configuration).
Allow extra disk space for environment and cache overhead. No API key needed.
The example dependencies do not become BOOTH runtime dependencies.
"""
from __future__ import annotations

import json
import platform
import re

try:
    import booth
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as exc:
    raise SystemExit(
        f"Missing example dependency ({exc.name}). Install boothpy, torch, "
        "transformers, and sentencepiece as shown in this script's docstring."
    ) from exc

MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"

# Intentionally different, uncalibrated defaults for differently scaled scores.
# Do not compare the numeric scores as if they were calibrated probabilities.
WEAK_THRESHOLD = 0.5
NLI_THRESHOLD = 0.7

# Illustrative development cases, not independent or held-out accuracy evidence.
CASES = [
    {
        "id": "exact_support",
        "answer": "Standard plans can be refunded within 30 days of purchase.",
        "evidence": ["Standard plans can be refunded within 30 days of the initial purchase."],
        "expected": "ACCEPTED",
        "control": "positive",
    },
    {
        "id": "paraphrase_support",
        "answer": "Customers with a Standard plan may obtain a refund during the first 30 days.",
        "evidence": ["Standard plans can be refunded within 30 days of the initial purchase."],
        "expected": "ACCEPTED",
        "control": "paraphrase-invariance",
    },
    {
        "id": "wrong_number",
        "answer": "Standard plans can be refunded within 90 days of the initial purchase.",
        "evidence": ["Standard plans can be refunded within 30 days of the initial purchase."],
        "expected": "BLOCKED",
        "control": "quantity-mutation",
    },
    {
        "id": "negation_flip",
        "answer": "Enterprise plans are refundable after activation.",
        "evidence": ["Enterprise plans are non-refundable after activation."],
        "expected": "BLOCKED",
        "control": "negation-mutation",
    },
    {
        "id": "scope_swap",
        "answer": "Standard plans are non-refundable after activation.",
        "evidence": ["Enterprise plans are non-refundable after activation."],
        "expected": "BLOCKED",
        "control": "entity-scope-mutation",
    },
    {
        "id": "cancellation_refund_inversion",
        "answer": "Cancelling an Annual plan at any time automatically creates a refund.",
        "evidence": ["Annual plans may be cancelled at any time, but cancellation does not automatically create a refund."],
        "expected": "BLOCKED",
        "control": "policy-negation-mutation",
    },
    {
        "id": "unrelated_evidence",
        "answer": "Standard plans can be refunded within 30 days.",
        "evidence": ["Technical support is available 24 hours a day."],
        "expected": "BLOCKED",
        "control": "irrelevant-evidence-negative",
    },
    {
        "id": "conflicting_evidence",
        "answer": "Standard plans can be refunded within 30 days of purchase.",
        "evidence": [
            "Standard plans can be refunded within 30 days of the initial purchase.",
            "Standard plans are non-refundable after purchase, including during the first 30 days.",
        ],
        "expected": "BLOCKED",
        "control": "conflict-negative",
    },
    {
        "id": "support_plus_irrelevant",
        "answer": "Standard plans can be refunded within 30 days of purchase.",
        "evidence": [
            "Standard plans can be refunded within 30 days of the initial purchase.",
            "Technical support is available 24 hours a day.",
        ],
        "expected": "ACCEPTED",
        "control": "irrelevant-evidence-invariance",
    },
    {
        "id": "duplicate_support",
        "answer": "Standard plans can be refunded within 30 days of purchase.",
        "evidence": [
            "Standard plans can be refunded within 30 days of the initial purchase.",
            "Standard plans can be refunded within 30 days of the initial purchase.",
        ],
        "expected": "ACCEPTED",
        "control": "duplicate-invariance",
    },
]



WORD_RE = re.compile(r"[A-Za-z0-9]+")


def lexical_overlap(answer: str, evidence: list[str]) -> float:
    """Fraction of unique answer tokens present in the evidence, ignoring meaning."""
    answer_words = set(WORD_RE.findall(answer.lower()))
    evidence_words = set(WORD_RE.findall(" ".join(evidence).lower()))
    return len(answer_words & evidence_words) / len(answer_words) if answer_words else 0.0


class PinnedDeBERTaComparator:
    """Passage-level three-way NLI, adapted to BOOTH's scalar compare_fn."""

    def __init__(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(0)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID, revision=REVISION, trust_remote_code=False
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_ID, revision=REVISION, trust_remote_code=False
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(
                f"Could not load pinned NLI model {MODEL_ID}@{REVISION}. "
                "Check network access, available disk space, and installed dependencies. "
                f"Original error: {exc}"
            ) from exc

        self.tokenizer = tokenizer
        self.model = model.to("cpu").eval()
        labels = [model.config.id2label[i].lower() for i in range(len(model.config.id2label))]
        expected_labels = {"entailment", "neutral", "contradiction"}
        if set(labels) != expected_labels:
            raise SystemExit(f"Unexpected NLI labels: {labels}; expected {expected_labels}")
        self.labels = labels
        self.indices = {label: index for index, label in enumerate(labels)}
        self.last_passages: list[dict] = []

    def __call__(self, answer: str, evidence: list[str]) -> float:
        self.last_passages = []
        for passage in evidence:
            # NLI order is premise (evidence), hypothesis (answer).
            tokens = self.tokenizer(passage, answer, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = self.model(**tokens).logits[0]
            probs = torch.softmax(logits, dim=-1)
            label = self.labels[int(torch.argmax(logits))]
            record = {
                "label": label,
                "p_entail": float(probs[self.indices["entailment"]]),
                "p_neutral": float(probs[self.indices["neutral"]]),
                "p_contradict": float(probs[self.indices["contradiction"]]),
            }
            self.last_passages.append(record)

        # Common simple adapter: a claim passes if any passage strongly entails it.
        # It intentionally does NOT resolve conflicting evidence. The conflict
        # control shows how this scalar projection can hide contradiction.
        return max((record["p_entail"] for record in self.last_passages), default=0.0)


def assess(result: booth.BoothResult, expected: str) -> dict:
    return {
        "status": result.status,
        "reason": result.reason,
        "detail": result.detail,
        "evidence_agreement": result.evidence_agreement,
        "matches_expected": result.status == expected,
    }


def summarize(rows: list[dict], arm: str) -> dict:
    return {
        "correct": sum(row[arm]["matches_expected"] for row in rows),
        "total": len(rows),
        "false_accept": sum(
            row["expected"] == booth.BLOCKED and row[arm]["status"] == booth.ACCEPTED
            for row in rows
        ),
        "false_block": sum(
            row["expected"] == booth.ACCEPTED and row[arm]["status"] == booth.BLOCKED
            for row in rows
        ),
        "uncertain": sum(row[arm]["status"] == booth.UNCERTAIN for row in rows),
    }


def main() -> None:
    print(
        "CONFIG "
        + json.dumps(
            {
                "python": platform.python_version(),
                "system": platform.system(),
                "machine": platform.machine(),
                "booth": booth.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "model": MODEL_ID,
                "model_revision": REVISION,
                "lexical_threshold": WEAK_THRESHOLD,
                "nli_threshold": NLI_THRESHOLD,
            },
            sort_keys=True,
        )
    )
    nli_compare = PinnedDeBERTaComparator()
    rows = []
    for case in CASES:
        weak_result = booth.check_with_evidence(
            answer=case["answer"],
            evidence=case["evidence"],
            compare_fn=lexical_overlap,
            evidence_threshold=WEAK_THRESHOLD,
        )
        nli_result = booth.check_with_evidence(
            answer=case["answer"],
            evidence=case["evidence"],
            compare_fn=nli_compare,
            evidence_threshold=NLI_THRESHOLD,
        )
        row = {
            "id": case["id"],
            "control": case["control"],
            "expected": case["expected"],
            "lexical": assess(weak_result, case["expected"]),
            "nli": assess(nli_result, case["expected"]),
            "nli_passages": list(nli_compare.last_passages),
        }
        rows.append(row)
        print("CASE " + json.dumps(row, sort_keys=True))

    print(
        "SUMMARY "
        + json.dumps(
            {
                "lexical": summarize(rows, "lexical"),
                "nli": summarize(rows, "nli"),
                "note": (
                    "Small author-designed controls; no source-disjoint evaluation, "
                    "model calibration, or factual-truth warrant."
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
