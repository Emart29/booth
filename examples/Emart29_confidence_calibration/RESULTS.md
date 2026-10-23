# BOOTH confidence calibration: does self-reported confidence predict correct answers?

## Question and scope

`booth.check()` accepts an answer when the model's self-reported `confidence >= 0.7`.
This example measures whether that number separates right answers from wrong ones
on a labelled QA set. It is a standalone example. It changes nothing in `src/booth/`
and does not implement `booth.measure()`.

Scope was posted on [issue #2](https://github.com/Vedantgitbot/booth/issues/2#issuecomment-6338193653);
related: [#21 `booth.measure()`](https://github.com/Vedantgitbot/booth/issues/21).

## Real run details

| Parameter | Value |
|---|---|
| **Date** | 2026-10-22 to 2026-10-23 |
| **BOOTH** (`pip show boothpy`) | `boothpy` 0.5.6 from PyPI |
| **Function used** | `check()` only, with `threshold=0.7`, `max_retries=1` (both the defaults, passed explicitly) and no `validator` |
| **Models** | `qwen2.5:7b-instruct` (digest `845dbda0ea48…697e`) and `llama3.2:3b` (digest `a80c4f17acd5…8b72`), via Ollama 0.40.8 |
| **Sampling** | `temperature=0`, `seed=0`, Ollama defaults otherwise (no output-length cap) |
| **Prompt** | the TriviaQA question verbatim. BOOTH appends its own JSON/confidence suffix, and there is no system prompt, context or retrieval |
| **Hardware** | Linux x86_64 cloud container, 4 vCPU (Intel Xeon), 15 GB RAM, **CPU only** |
| **Python** | CPython 3.13.16 |
| **Runs** | 1 run per model over the same 300 questions: 600 `check()` calls and 613 model calls |
| **Wall time** | qwen2.5:7b took 2,723 s (median 8.1 s per question). llama3.2:3b took 5,014 s (median 4.3 s), of which 3,600 s were three 2×600 s timeouts |
| **Hosted model** | Groq `llama-3.3-70b-versatile` is supported by the script but **not run**, because no `GROQ_API_KEY` was available |

Full model digests are in the `CONFIG` lines under [Actual stdout](#actual-stdout).

### Dataset

[TriviaQA](https://nlp.cs.washington.edu/triviaqa/) (Joshi, Choi, Weld and Zettlemoyer, ACL 2017),
`rc.nocontext` validation split, from the [`mandarjoshi/trivia_qa`](https://huggingface.co/datasets/mandarjoshi/trivia_qa)
parquet at revision `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f`. It is licensed Apache 2.0
([upstream LICENSE](https://github.com/mandarjoshi90/triviaqa/blob/master/LICENSE)).

- **300 questions**, drawn with `random.Random(0).sample(...)` from the split's **9,960 unique `question_id`s**.
  The split has 17,944 rows because many questions are listed once per evidence domain (web and Wikipedia).
  My first attempt sampled rows, which over-weighted those questions and drew one question twice. I discarded
  that run and resampled by unique id. All numbers here come from the deduplicated sample.
- `triviaqa_sample.json` holds the id, question, canonical value and answer aliases for each question.
  `python run_example.py --build-sample <parquet>` rebuilds it byte for byte.

### How "correct" is decided

1. **Grader (deterministic):** SQuAD-style normalisation (lowercase, strip punctuation and the articles
   a/an/the, collapse whitespace) is applied to BOOTH's `answer` and to every TriviaQA alias. An answer is
   **correct** if it equals an alias or contains one as a whole-word span. No LLM judge is used.
2. **Calibration is measured on attempt 0** (`result.attempts[0]`). That is the confidence the 0.7 gate
   sees first; a retried answer's confidence comes from a different prompt. Items with no usable attempt-0
   confidence (call failed, parse failure or `AMBIGUOUS`) are excluded from calibration and **counted**.
3. **Grader audit:** for each model, a fixed random sample of 50 items (`random.Random(1)`, the same ids for
   both models) was checked by hand against the grader. See [Grader audit](#grader-audit).

## Summary

| Attempt 0, usable answers | qwen2.5:7b-instruct | llama3.2:3b |
|---|---:|---:|
| Usable / 300 | 288 | 291 |
| Accuracy | 149/288 = **51.7%** [46–57%] | 148/291 = **50.9%** [45–57%] |
| Confidence values the model emitted | 0.8, 0.9, 0.95, 1.0 | 0, 0.2, 0.8, 0.9, 1.0 |
| Answers clearing 0.7 | 288/288 (100%) | 283/291 (97.3%) |
| **Wrong answers that cleared 0.7** | **139 of 139 wrong (100%)** | **135 of 143 wrong (94.4%)** |
| Share of answers clearing 0.7 that were wrong | 139/288 = 48.3% | 135/283 = 47.7% |
| Correct answers held back by 0.7 | 0 | 0 |
| AUROC of confidence (0.5 = no ranking signal) | 0.637 | 0.679 |
| Brier score (base-rate predictor ≈ 0.250) | 0.405 | 0.329 |
| ECE (10 equal-width bins) | 0.404 | 0.311 |

Intervals are Wilson 95%.

**Main finding.** On this sample, the default 0.7 gate almost never rejected a wrong answer.
Neither model ever reported a confidence between 0.2 and 0.8. qwen2.5:7b put every
usable answer at 0.8 or above. llama3.2:3b's only answers below 0.7 were eight cases
where it effectively abstained: seven `"None"` answers at confidence 0.0, and one wrong
answer at 0.2. About half of all accepted answers were wrong.

**There is some ranking signal in the confidence, but not at 0.7.** Within each model, the higher value it
emits is more often right:

| Model | Lower value | Higher value |
|---|---|---|
| llama3.2:3b | 0.8 → 64/166 = 38.6% correct | 0.9 → 84/116 = 72.4% correct |
| qwen2.5:7b-instruct | 0.9 → 69/165 = 41.8% correct | 0.95 → 71/103 = 68.9% correct (1.0 → 9/15 = 60.0%) |

So the useful cut point is model-specific: 0.85 for llama3.2:3b and 0.95 for qwen2.5:7b. At those
thresholds, accepted accuracy rises to 71.8% at 40.2% coverage (llama) and 67.8% at 41.0% coverage (qwen),
and about a third of accepted answers are still wrong. The same number means different things
for different models: 0.9 was the *higher* tier for llama and the *lower* tier for qwen.
None of this was tuned on held-out data, and these thresholds were read off the same 300 questions.

**The default retry hardly ever ran.** Reconsideration only happens below the threshold, and almost
nothing fell below it. There were 13 retries in 600 checks: 12 for llama (mostly the abstentions, plus
the 3 timeouts) and 1 for qwen (a parse failure). Only one wrong answer was fixed (llama, wrong→right),
and no right answer was talked out of being right. This sample therefore says almost nothing about
whether reconsideration helps. It mostly shows that, at 0.7, reconsideration is almost never triggered.

**`AMBIGUOUS` is a separate path and behaved differently.** qwen flagged 11 of 300 questions as ambiguous
(2 of those answers were correct) and llama flagged 5 (3 correct). Several of qwen's "interpretations"
were candidate answers rather than readings of the question. For example, it listed `["Panama", "Somalia"]`
for an election question, and `["A dancer was Oscar-nominated…", "The Towering Inferno is a film, not a person"]`.

## Other observations

- **Abstention is the only low-confidence behaviour seen.** llama3.2:3b's seven confidence-0.0 answers were
  all the literal string `"None"`, and BOOTH returned `UNCERTAIN` for them. In practice that is the gate's only
  useful rejection in this run, and it comes from the model abstaining, not from graded confidence.
- **Timeouts.** Three llama3.2:3b questions hit the script's 600 s HTTP timeout on both attempts (`call_failed=True`,
  `UNCERTAIN`). No output-length cap was set, so runaway generation is the likely cause. Nothing came back,
  so this can't be confirmed. `result.call_failed` correctly told these apart from model answers.
- **Language switch.** qwen2.5:7b answered one question with `氧` (Chinese for "oxygen") at confidence 0.9.
  The correct answer was silicon, so it was wrong either way.
- **Parse failures** were rare: 1 per model. BOOTH's parse-failure retry recovered qwen's.

## Grader audit

The same 50 ids were checked for both models (100 items). I disagreed with the grader on **4/100 (4%)**,
and every disagreement was the grader **under-crediting** a correct answer:

| Model | id | Answer | TriviaQA alias(es) | Grader | My label |
|---|---|---|---|---|---|
| llama3.2:3b | `qw_15046` | "a depression or groove in the ground" | "Narrow trench" | wrong | correct (paraphrase) |
| llama3.2:3b | `sfq_5282` | "a member of the family Corvidae, which includes crows, ravens, jays, and magpies" | "Aves", "bird", … | wrong | correct (answer never says "bird") |
| qwen2.5:7b | `qw_15046` | "A groove or channel made in soil by raindrops, wind, or the passage of plows…" | "Narrow trench" | wrong | correct (paraphrase) |
| qwen2.5:7b | `qw_1282` | "Prussian War of 1870-1871" | "Franco-Prussian War", "1870 War", … | wrong | correct |

If the audit rate held across the whole sample, the true accuracies would be a few points higher than reported.
That is too small to change the main finding: even crediting all four, most wrong answers still clear 0.7.
Some dataset noise was also kept as-is. For example, `qg_4484` ("What radio station do you find at 90.3 on your FM dial?")
has no single right answer.

The audit was a single reviewer's judgement and is not independent. The full 50-item tables come from
`python run_example.py --report-only --audit 50`.

## Caveats

- **One dataset, one prompt format, n = 300 per model, one run each.** Trivia recall from a 3B or 7B model
  on CPU isn't representative of every workload or of larger models.
- **Temperature 0** measures one deterministic sample per question, not the model's answer distribution.
  Byte-identical outputs on other hardware or Ollama versions are not guaranteed.
- **The thresholds discussed above (0.85 and 0.95) were picked by looking at these results.** They are
  descriptive, not validated.
- **The grader is lenient on containment** (an answer that *contains* an alias counts as correct) and strict
  on paraphrase. The audit measures the second effect but cannot rule out every false positive.
- **ECE and Brier** are dominated by the fact that nearly all confidences sit at 0.8 to 0.95 while accuracy
  is about 51%. With so few distinct values, the bucket table and the per-value accuracies are more informative.
- These results say nothing about `check_with_evidence()`, which does not use self-reported confidence.

## Reproducing

```bash
pip install boothpy==0.5.6
ollama pull qwen2.5:7b-instruct && ollama pull llama3.2:3b
python run_example.py                                # ~1–2 h on 4 CPU cores; resumable
python run_example.py --report-only --audit 50       # tables + audit sample
```

- `run_example.py` only uses the standard library plus `boothpy`. Model calls go to Ollama over HTTP with `urllib`.
- `raw_outputs_qwen2.5-7b-instruct.jsonl.gz` and `raw_outputs_llama3.2-3b.jsonl.gz` hold, one file per model, every
  `BoothResult.to_dict()` (all attempts, `raw_text`, parsed fields) plus grades and timings for the run above.
  `--report-only` reads them if no local uncompressed `raw_outputs_<model>.jsonl` exists. A fresh run writes its own
  `raw_outputs_<model>.jsonl` and never resumes from the committed files.
- **Determinism spot check:** re-running the first sampled question (`qw_8575`) for llama3.2:3b from a fresh copy,
  with only the committed files present, made a new model call and reproduced the committed `raw_text` byte for byte
  on the same machine.
- **After the model run, the script was edited in ways that don't affect what the models were asked:**
  it now prints the per-value accuracy line, `--report-only` can read the `.gz` files, resume no longer reads the
  committed files, and raw outputs are split into one file per model (the rows themselves are unchanged). The report
  below was regenerated from the saved outputs with the final script, and is byte-identical whether it is read from
  the `.jsonl` or the `.jsonl.gz` files.

| File | SHA-256 |
|---|---|
| `run_example.py` (final) | `9ad8182f31971351a6c4488e35721aa51ce9e897c2a2d3936a45a8ccfa8fe51c` |
| `triviaqa_sample.json` | `ef6ded880d36b1a7b0f46444e99c0cf5e52b8c21e27ef39a9fe502e260654618` |
| `raw_outputs_qwen2.5-7b-instruct.jsonl` (uncompressed, 300 lines) | `2d0debbe8e74a79e418fe510cf2b63f892a69df7b42ffd14ade00760d9806747` |
| `raw_outputs_llama3.2-3b.jsonl` (uncompressed, 300 lines) | `5759d74bd5efb1d88ac71a4ba9423a05b22255a652f6a3d48fb4c8d588a6a374` |
| report stdout below | `a3d670947cfa1b66d3a498423c48023d72afa8ab8b45e37faa1580b68c4ab710` |

## Actual stdout

The `CONFIG` lines printed when each model was run:

```text
CONFIG {"booth": "0.5.6", "groq": "skipped (GROQ_API_KEY not set)", "max_retries": 1, "models": {"llama3.2:3b": "a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72"}, "ollama": "0.40.8", "python": "3.13.16", "questions": 300, "seed": 0, "system": "Linux x86_64", "temperature": 0.0, "threshold": 0.7}
CONFIG {"booth": "0.5.6", "groq": "skipped (GROQ_API_KEY not set)", "max_retries": 1, "models": {"qwen2.5:7b-instruct": "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e"}, "ollama": "0.40.8", "python": "3.13.16", "questions": 300, "seed": 0, "system": "Linux x86_64", "temperature": 0.0, "threshold": 0.7}
```

The complete output of `python run_example.py --report-only`:

```text
CONFIG {"booth": "0.5.6", "max_retries": 1, "python": "3.13.16", "questions": 300, "seed": 0, "system": "Linux x86_64", "temperature": 0.0, "threshold": 0.7}

## qwen2.5:7b-instruct

Questions: 300. Usable attempt-0 answers (parsed, not ambiguous, call ok): 288.
Excluded from calibration: call_failed=0, parse_failure=1, ambiguous=11.
Attempt-0 accuracy on usable answers: 149/288 = 51.7% [46–57%].
Wall time per question (incl. retries): median 8.1s.

### Accuracy by attempt-0 confidence bucket

| Confidence | n | Correct | Accuracy [95% Wilson] |
|---|---:|---:|---|
| [0.0, 0.5) | 0 | 0 | n/a |
| [0.5, 0.7) | 0 | 0 | n/a |
| [0.7, 0.8) | 0 | 0 | n/a |
| [0.8, 0.9) | 5 | 0 | 0.0% [0–43%] |
| [0.9, 1.0) | 268 | 140 | 52.2% [46–58%] |
| = 1.0 | 15 | 9 | 60.0% [36–80%] |

Accuracy per distinct confidence value emitted: 0.8 → 0/5 (0.0%), 0.9 → 69/165 (41.8%), 0.95 → 71/103 (68.9%), 1 → 9/15 (60.0%)

### The 0.7 gate on attempt 0

| | Value |
|---|---|
| Accuracy, confidence ≥ 0.7 | 149/288 = 51.7% [46–57%] |
| Accuracy, confidence < 0.7 | 0/0 = n/a |
| Wrong answers that cleared 0.7 | 139 |
| ...as share of all wrong answers (missed by gate) | 139/139 = 100.0% |
| ...as share of all answers clearing 0.7 (false-accept rate) | 139/288 = 48.3% |
| Correct answers held back (< 0.7) | 0/149 = 0.0% of correct |

### Threshold sweep (attempt 0 only)

| Threshold | Coverage | Accuracy of accepted | Wrong accepted |
|---:|---:|---:|---:|
| 0.50 | 288/288 = 100.0% | 51.7% | 139 |
| 0.60 | 288/288 = 100.0% | 51.7% | 139 |
| 0.70 | 288/288 = 100.0% | 51.7% | 139 |
| 0.80 | 288/288 = 100.0% | 51.7% | 139 |
| 0.85 | 283/288 = 98.3% | 52.7% | 134 |
| 0.90 | 283/288 = 98.3% | 52.7% | 134 |
| 0.95 | 118/288 = 41.0% | 67.8% | 38 |
| 1.00 | 15/288 = 5.2% | 60.0% | 6 |

### Threshold-free summaries (attempt 0)

- AUROC (0.5 = confidence doesn't rank correct above wrong): 0.637
- Brier score (lower is better; always predicting base rate gives 0.250): 0.405
- ECE, 10 equal-width bins: 0.404

### What the default retry (max_retries=1) did

Final statuses: ACCEPTED=288, AMBIGUOUS=11, REPAIRED=1
- Retried: 1. Came back REPAIRED: 1, of which correct: 0.
- Attempt 0 → attempt 1: wrong→right 0, right→wrong 0, right→right 0, wrong→wrong 1.
- End to end, result.ok answers: 289/300; correct 149/289 = 51.6%; wrong but ok: 140.

## llama3.2:3b

Questions: 300. Usable attempt-0 answers (parsed, not ambiguous, call ok): 291.
Excluded from calibration: call_failed=3, parse_failure=1, ambiguous=5.
Attempt-0 accuracy on usable answers: 148/291 = 50.9% [45–57%].
Wall time per question (incl. retries): median 4.3s.

### Accuracy by attempt-0 confidence bucket

| Confidence | n | Correct | Accuracy [95% Wilson] |
|---|---:|---:|---|
| [0.0, 0.5) | 8 | 0 | 0.0% [0–32%] |
| [0.5, 0.7) | 0 | 0 | n/a |
| [0.7, 0.8) | 0 | 0 | n/a |
| [0.8, 0.9) | 166 | 64 | 38.6% [31–46%] |
| [0.9, 1.0) | 116 | 84 | 72.4% [64–80%] |
| = 1.0 | 1 | 0 | 0.0% [0–79%] |

Accuracy per distinct confidence value emitted: 0 → 0/7 (0.0%), 0.2 → 0/1 (0.0%), 0.8 → 64/166 (38.6%), 0.9 → 84/116 (72.4%), 1 → 0/1 (0.0%)

### The 0.7 gate on attempt 0

| | Value |
|---|---|
| Accuracy, confidence ≥ 0.7 | 148/283 = 52.3% [46–58%] |
| Accuracy, confidence < 0.7 | 0/8 = 0.0% [0–32%] |
| Wrong answers that cleared 0.7 | 135 |
| ...as share of all wrong answers (missed by gate) | 135/143 = 94.4% |
| ...as share of all answers clearing 0.7 (false-accept rate) | 135/283 = 47.7% |
| Correct answers held back (< 0.7) | 0/148 = 0.0% of correct |

### Threshold sweep (attempt 0 only)

| Threshold | Coverage | Accuracy of accepted | Wrong accepted |
|---:|---:|---:|---:|
| 0.50 | 283/291 = 97.3% | 52.3% | 135 |
| 0.60 | 283/291 = 97.3% | 52.3% | 135 |
| 0.70 | 283/291 = 97.3% | 52.3% | 135 |
| 0.80 | 283/291 = 97.3% | 52.3% | 135 |
| 0.85 | 117/291 = 40.2% | 71.8% | 33 |
| 0.90 | 117/291 = 40.2% | 71.8% | 33 |
| 0.95 | 1/291 = 0.3% | 0.0% | 1 |
| 1.00 | 1/291 = 0.3% | 0.0% | 1 |

### Threshold-free summaries (attempt 0)

- AUROC (0.5 = confidence doesn't rank correct above wrong): 0.679
- Brier score (lower is better; always predicting base rate gives 0.250): 0.329
- ECE, 10 equal-width bins: 0.311

### What the default retry (max_retries=1) did

Final statuses: ACCEPTED=283, AMBIGUOUS=5, REPAIRED=2, UNCERTAIN=10
- Retried: 12. Came back REPAIRED: 2, of which correct: 1.
- Attempt 0 → attempt 1: wrong→right 1, right→wrong 0, right→right 0, wrong→wrong 11.
- End to end, result.ok answers: 285/300; correct 149/285 = 52.3%; wrong but ok: 136.
```
