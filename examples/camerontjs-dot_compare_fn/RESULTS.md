# BOOTH comparator example: lexical overlap vs pinned DeBERTa NLI

## Question and scope

Does choosing a semantic natural-language-inference (NLI) `compare_fn` instead of a weak
word-overlap `compare_fn` change the acceptance decisions returned by BOOTH for the
**same fixed claims and evidence**?

This is an example for [issue #18](https://github.com/Vedantgitbot/booth/issues/18),
related to [provider example issue #2](https://github.com/Vedantgitbot/booth/issues/2).
No BOOTH source code, dependency declarations, or public behavior changes are proposed.

## Real run details

- **Date:** 2026-10-19.
- **Upstream BOOTH source base:** `dfe29eb95406bce9822d40cf36babe9e6b753c26`.
- **BOOTH package:** `boothpy==0.5.4` from PyPI, verified with `pip show boothpy`.
- **Python / OS:** CPython 3.9.6, macOS arm64; local CPU inference.
- **Dependencies for this example only:** torch 2.8.0, transformers 4.57.6, sentencepiece 0.2.2.
- **NLI model:** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`.
- **Functions:** `booth.check_with_evidence()`; neither BOOTH `check()` nor an LLM generator is involved.
- **Runs:** 2 identical executions on 10 fixed, authored claim/evidence cases; no retries or model-generated answers.
- **Script SHA-256:** `f5079fdbdd2870fa4e563eb21cc2bfa2a49be7a1f6d0b8097ef1967703e0fb93`.
- **Standard output SHA-256, both runs:** `271bec75f9d7bd435e5b3eaf5b5801f5b0b7fa1c2559a0d9bbaf13f0712820e2`.

The answer and evidence sequence for each case are held constant between comparator arms.
The lexical checker counts unique matching words and accepts at a **0.5 overlap threshold**.
DeBERTa scores each passage as a premise against the complete claim as hypothesis,
then passes its **maximum entailment softmax score** to BOOTH at a **0.7 threshold**.
Both thresholds were selected for this demonstration, **not independently calibrated**.
Scores from the two methods do not represent equivalent probability scales.

All cases are hand-authored demonstrations, not a held-out or source-disjoint evaluation.
Expected `BLOCKED` means *not acceptable under this fixture's intended checkpoint*;
it does **not** mean the evidence logically refutes the claim. An unrelated source, for
example, justifies no acceptance, not a contradiction verdict.

## Summary

| Outcome against the authored expected checkpoint status | Weak lexical | Pinned DeBERTa |
|---|---:|---:|
| Cases matching expected status | 4 / 10 | 9 / 10 |
| False accepts (expected blocked, accepted) | 5 | 1 |
| False blocks (expected accepted, blocked) | 1 | 0 |
| Uncertain | 0 | 0 |

The lexical checker accepts a wrong **90-day** refund window because nearly every word
overlaps a 30-day rule; accepts flipped refund negation and an Enterprise/Standard
scope swap; and rejects a supported paraphrase with different vocabulary.
DeBERTa's checkpoint decisions are better on these examples.

The important DeBERTa-side failure is **conflicting evidence**:
one passage has `p(entail)=0.993838`, while another has
`p(contradict)=0.996633`. DeBERTa *identified both relations*, but the
simple `max(p_entail)` comparator discarded contradiction information,
so BOOTH received a high score and returned `ACCEPTED`.
This is an adapter/aggregation limitation, **not a BOOTH core defect**.

Another limitation appears in the **scope swap** case. The evidence is about
Enterprise plans but the claim is about Standard plans. DeBERTa assigns the
passage its top label `contradiction` (about 0.752), although the evidence
does not logically refute an unrelated Standard-plan rule. BOOTH correctly
does not accept the claim in this fixture, but that does **not** validate
DeBERTa's finer contradiction labeling.

The current comparison provides an illustrative **capability contrast**.
It does not establish general comparator accuracy, model calibration, actual
source legitimacy, corpus sufficiency, or successful CAL V1 integration.

## Actual stdout from run 1

The following is the complete output printed by `run_example.py` (stdout,
not manually recomposed predictions). Run 2 produced the same stdout bytes.

```text
CONFIG {"booth": "0.5.4", "lexical_threshold": 0.5, "machine": "arm64", "model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "model_revision": "6f5cf0a2b59cabb106aca4c287eed12e357e90eb", "nli_threshold": 0.7, "python": "3.9.6", "system": "Darwin", "torch": "2.8.0", "transformers": "4.57.6"}
CASE {"control": "positive", "expected": "ACCEPTED", "id": "exact_support", "lexical": {"detail": null, "evidence_agreement": 1.0, "matches_expected": true, "reason": null, "status": "ACCEPTED"}, "nli": {"detail": null, "evidence_agreement": 0.993838369846344, "matches_expected": true, "reason": null, "status": "ACCEPTED"}, "nli_passages": [{"label": "entailment", "p_contradict": 0.0009218579507432878, "p_entail": 0.993838369846344, "p_neutral": 0.005239792633801699}]}
CASE {"control": "paraphrase-invariance", "expected": "ACCEPTED", "id": "paraphrase_support", "lexical": {"detail": "score 0.3076923076923077 below evidence_threshold 0.5", "evidence_agreement": 0.3076923076923077, "matches_expected": false, "reason": "EVIDENCE_DISAGREES", "status": "BLOCKED"}, "nli": {"detail": null, "evidence_agreement": 0.9941997528076172, "matches_expected": true, "reason": null, "status": "ACCEPTED"}, "nli_passages": [{"label": "entailment", "p_contradict": 0.00045581915765069425, "p_entail": 0.9941997528076172, "p_neutral": 0.005344494711607695}]}
CASE {"control": "quantity-mutation", "expected": "BLOCKED", "id": "wrong_number", "lexical": {"detail": null, "evidence_agreement": 0.9166666666666666, "matches_expected": false, "reason": null, "status": "ACCEPTED"}, "nli": {"detail": "score 0.001540952012874186 below evidence_threshold 0.7", "evidence_agreement": 0.001540952012874186, "matches_expected": true, "reason": "EVIDENCE_DISAGREES", "status": "BLOCKED"}, "nli_passages": [{"label": "contradiction", "p_contradict": 0.995970606803894, "p_entail": 0.001540952012874186, "p_neutral": 0.0024884925223886967}]}
CASE {"control": "negation-mutation", "expected": "BLOCKED", "id": "negation_flip", "lexical": {"detail": null, "evidence_agreement": 1.0, "matches_expected": false, "reason": null, "status": "ACCEPTED"}, "nli": {"detail": "score 0.0007738820859231055 below evidence_threshold 0.7", "evidence_agreement": 0.0007738820859231055, "matches_expected": true, "reason": "EVIDENCE_DISAGREES", "status": "BLOCKED"}, "nli_passages": [{"label": "contradiction", "p_contradict": 0.9982706308364868, "p_entail": 0.0007738820859231055, "p_neutral": 0.0009555029100738466}]}
CASE {"control": "entity-scope-mutation", "expected": "BLOCKED", "id": "scope_swap", "lexical": {"detail": null, "evidence_agreement": 0.8571428571428571, "matches_expected": false, "reason": null, "status": "ACCEPTED"}, "nli": {"detail": "score 0.07546699792146683 below evidence_threshold 0.7", "evidence_agreement": 0.07546699792146683, "matches_expected": true, "reason": "EVIDENCE_DISAGREES", "status": "BLOCKED"}, "nli_passages": [{"label": "contradiction", "p_contradict": 0.751786470413208, "p_entail": 0.07546699792146683, "p_neutral": 0.17274649441242218}]}
CASE {"control": "policy-negation-mutation", "expected": "BLOCKED", "id": "cancellation_refund_inversion", "lexical": {"detail": null, "evidence_agreement": 0.6363636363636364, "matches_expected": false, "reason": null, "status": "ACCEPTED"}, "nli": {"detail": "score 0.0005258059245534241 below evidence_threshold 0.7", "evidence_agreement": 0.0005258059245534241, "matches_expected": true, "reason": "EVIDENCE_DISAGREES", "status": "BLOCKED"}, "nli_passages": [{"label": "contradiction", "p_contradict": 0.9978524446487427, "p_entail": 0.0005258059245534241, "p_neutral": 0.001621777075342834}]}
CASE {"control": "irrelevant-evidence-negative", "expected": "BLOCKED", "id": "unrelated_evidence", "lexical": {"detail": "score 0.0 below evidence_threshold 0.5", "evidence_agreement": 0.0, "matches_expected": true, "reason": "EVIDENCE_DISAGREES", "status": "BLOCKED"}, "nli": {"detail": "score 0.0009648687555454671 below evidence_threshold 0.7", "evidence_agreement": 0.0009648687555454671, "matches_expected": true, "reason": "EVIDENCE_DISAGREES", "status": "BLOCKED"}, "nli_passages": [{"label": "neutral", "p_contradict": 0.0003691909369081259, "p_entail": 0.0009648687555454671, "p_neutral": 0.9986659288406372}]}
CASE {"control": "conflict-negative", "expected": "BLOCKED", "id": "conflicting_evidence", "lexical": {"detail": null, "evidence_agreement": 1.0, "matches_expected": false, "reason": null, "status": "ACCEPTED"}, "nli": {"detail": null, "evidence_agreement": 0.993838369846344, "matches_expected": false, "reason": null, "status": "ACCEPTED"}, "nli_passages": [{"label": "entailment", "p_contradict": 0.0009218579507432878, "p_entail": 0.993838369846344, "p_neutral": 0.005239792633801699}, {"label": "contradiction", "p_contradict": 0.9966332316398621, "p_entail": 0.0006440724246203899, "p_neutral": 0.0027227369137108326}]}
CASE {"control": "irrelevant-evidence-invariance", "expected": "ACCEPTED", "id": "support_plus_irrelevant", "lexical": {"detail": null, "evidence_agreement": 1.0, "matches_expected": true, "reason": null, "status": "ACCEPTED"}, "nli": {"detail": null, "evidence_agreement": 0.993838369846344, "matches_expected": true, "reason": null, "status": "ACCEPTED"}, "nli_passages": [{"label": "entailment", "p_contradict": 0.0009218579507432878, "p_entail": 0.993838369846344, "p_neutral": 0.005239792633801699}, {"label": "neutral", "p_contradict": 0.0006168576655909419, "p_entail": 0.0011228114599362016, "p_neutral": 0.9982603192329407}]}
CASE {"control": "duplicate-invariance", "expected": "ACCEPTED", "id": "duplicate_support", "lexical": {"detail": null, "evidence_agreement": 1.0, "matches_expected": true, "reason": null, "status": "ACCEPTED"}, "nli": {"detail": null, "evidence_agreement": 0.993838369846344, "matches_expected": true, "reason": null, "status": "ACCEPTED"}, "nli_passages": [{"label": "entailment", "p_contradict": 0.0009218579507432878, "p_entail": 0.993838369846344, "p_neutral": 0.005239792633801699}, {"label": "entailment", "p_contradict": 0.0009218579507432878, "p_entail": 0.993838369846344, "p_neutral": 0.005239792633801699}]}
SUMMARY {"lexical": {"correct": 4, "false_accept": 5, "false_block": 1, "total": 10, "uncertain": 0}, "nli": {"correct": 9, "false_accept": 1, "false_block": 0, "total": 10, "uncertain": 0}, "note": "Small author-designed controls; no source-disjoint evaluation, model calibration, or factual-truth warrant."}
```

Both runs exited 0. Their stderr contained a local Python LibreSSL/urllib3
compatibility warning and a transformers warning that the tokenizer has no
predefined maximum sequence length. Neither warning changed this short-fixture
run. Long-input truncation and source-context loss are **not evaluated** here.

## Reproduction

From a clean BOOTH checkout at `dfe29eb95406bce9822d40cf36babe9e6b753c26`
with this three-file example folder added:

```bash
python -m pip install boothpy==0.5.4 torch==2.8.0 transformers==4.57.6 sentencepiece==0.2.2
python examples/camerontjs-dot_compare_fn/run_example.py
```

The first run downloads the pinned model and tokenizer unless already cached.
An API key is not needed. The example itself adds no dependency to BOOTH.
The script reports missing packages and model-load failures explicitly.

The **existing upstream pytest suite** was run from this BOOTH checkout
under the same environment: **348 passed**. The change touches only files
inside `examples/camerontjs-dot_compare_fn/`.

## What this example teaches

BOOTH is a checkpoint wrapper: its `ACCEPTED` and `BLOCKED` reflect
the comparator the application supplies. A stronger semantic comparator
can improve useful acceptance and rejection on fixed claims, while still
misreading scope or hiding a material conflicting passage if the adapter
collapses richer outputs to a scalar. A valid comparison result is not proof
of objective truth. The minimum next check before treating an NLI-based
comparator as dependable would be fresh, independently adjudicated claims
with measured false acceptances, false blocks, neutral cases, and conflicts.
