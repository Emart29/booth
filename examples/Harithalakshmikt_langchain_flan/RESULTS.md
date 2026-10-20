# BOOTH + LangChain + FAISS + FLAN-T5-base

## Setup
- **Generator:** google/flan-t5-base, run locally via transformers (no API key)
- **Retrieval:** LangChain + FAISS, embeddings from sentence-transformers/all-MiniLM-L6-v2
- **Evidence comparison:** cross-encoder/nli-deberta-v3-base
- **BOOTH function used:** `check_with_evidence()` with `evidence_threshold=0.70`
- **Date run:** 2026-10-19
- **BOOTH version:** boothpy 0.5.4
- **Other versions:** langchain-text-splitters 1.1.3, langchain-community 0.4.2, faiss-cpu 1.15.1, transformers 5.20.0, sentence-transformers 6.1.0, torch 2.14.1
- **Python / OS:** Python 3.11.0 / Windows (version 26H2)

## What the example does
Five short library-policy sentences are defined inline and indexed in FAISS. For a question, the top-3 chunks are retrieved and FLAN-T5 answers from them. BOOTH then checks that answer against the same retrieved chunks using my own NLI-based `compare_fn`. After that, the script checks two hardcoded answers (one wrong, one correct) to the default question "What time does the library close on weekdays?", whatever question was typed, to see how the checker treats a simulated hallucination and a correct full sentence.

BOOTH only validates here. The answer comes from FLAN-T5 (or is hardcoded), and the score comes from my `compare_fn`.

## compare_fn used
An NLI cross-encoder scores whether the evidence entails each sentence of the answer.

1. The answer is split into sentences (regex on `.`, `!`, `?`), and each sentence is a claim.
2. Each claim is paired with each evidence passage (premise = evidence, hypothesis = claim).
3. Entailment probability is computed with a softmax over the logits.
4. Each claim takes its best score across passages, and the final score is the lowest claim score.

**Weak spots:**
- Entailment is an estimate of support, not proof of truth.
- NLI models expect full-sentence hypotheses. Short answers like "No" or "9 AM to 5 PM" score near zero.
- The question is never given to the NLI model, so a fragment loses the context that gives it meaning.
- Claims that need a small inference (closing time from opening hours) are scored low.
- Sentence splitting is naive.
- The 0.70 threshold was not tuned, and only a few checks were run.

## Runs
| # | Question | Answer | Agreement | Status | Answer quality | Expected? |
|---|----------|--------|-----------|--------|----------------|-----------|
| 1 | What time does the library close on weekdays? | 9 AM to 5 PM (FLAN-T5) | 0.0013 | BLOCKED | Imprecise: gives opening hours, not the closing time | Debatable |
| 2 | How many books can a member borrow? | up to five (FLAN-T5) | 0.882 | ACCEPTED | Correct | Yes |
| 3 | Can members borrow ten books at a time? | No (FLAN-T5) | 0.0047 | BLOCKED | Correct | **No (false block)** |
| 4 | What time does the library close on weekdays? | The library closes at 9 PM on weekdays. (hardcoded, simulates a hallucination) | 0.0043 | BLOCKED | Wrong | Yes |
| 5 | What time does the library close on weekdays? | The library closes at 5 PM on weekdays. (hardcoded control) | 0.168 | BLOCKED | Correct | **No (false block)** |

**Total:** 5 checks: 1 accepted, 4 blocked. The deliberately wrong answer was blocked as intended. Two correct answers (runs 3 and 5) were also blocked. No wrong answer was accepted, and the only acceptance was a correct one. Scores were identical across repeated runs of the same input, so scoring is deterministic.

## What worked well
- `check_with_evidence()` was easy to connect to a local model, and the `compare_fn` contract is simple to satisfy.
- Status, score and detail make it clear why an answer was blocked (for example `score 0.0013 below evidence_threshold 0.7`).
- The deliberately wrong answer was BLOCKED, and "up to five" was ACCEPTED at 0.88.
- The scores rank the correct sentence above the wrong one (0.168 vs 0.0043), so the NLI model has some signal.
- Everything runs locally with no API key.

## Problems and edge cases
- **Over-strict compare_fn.** At threshold 0.70, only a near-verbatim answer ("up to five", 0.88) passed. A correct full sentence that needs a small inference scored 0.168 and was blocked. A likely cause is that the evidence says "open from 9 AM to 5 PM" while the claim says "closes at 5 PM". Another possible cause is the five-sentence premise diluting the signal. Neither was tested.
- **Terse answers.** FLAN-T5 tends to answer with fragments. "No" (run 3) is correct, but no passage entails a bare "No", so it was blocked.
- **Generator imprecision (run 1).** FLAN-T5 returned the opening hours instead of the closing time. BOOTH blocked it, but the score (0.0013) is almost the same as for the wrong sentence (0.0043). The `compare_fn` penalizes any fragment, so it cannot tell an imprecise answer from a wrong one, and the block was not due to the checker understanding the problem.
- **Weak separation at the low end.** The scores are ordered correctly, but at a 0.70 threshold a score of 0.168 and a score of 0.004 get the same BLOCKED outcome.
- **Single evidence passage.** k=3 returned one passage in every run, because the whole knowledge base fits in one 300-character chunk.
- **Tiny sample.** 5 checks on one 5-sentence knowledge base. Nothing here supports a claim about general accuracy.
- **Not tried:** scoring against single sentences instead of whole chunks, rewriting short answers into full sentences before comparing, and tuning the threshold. I did not lower the threshold, to avoid tuning to a handful of examples.
- The "Decision" line printed by the script is my own wording, not a BOOTH field.

## Tips for reproducing
- Install and run commands are in the `run_example.py` docstring.
- The first run downloads FLAN-T5-base, MiniLM and the NLI model, so it needs internet and some disk space.
- Press Enter at the prompt to use the default question ("What time does the library close on weekdays?"), or type your own.
- If correct short answers are blocked, try rewriting them as full sentences before comparing, using sentence-level premises, or adjusting the threshold.