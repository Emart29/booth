"""
BOOTH evidence-checking example with a local RAG pipeline.

Provider/model:
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Answer generation: google/flan-t5-base
- Evidence comparison: cross-encoder/nli-deberta-v3-base

Run from the BOOTH repository root:
    pip install -e .
    pip install langchain-text-splitters langchain-community \
        sentence-transformers transformers torch faiss-cpu
    python examples/Harithalakshmikt_langchain_flan/run_example.py

No API key is required. The Hugging Face models are downloaded
on first use and then run locally.

After your question is answered, the script also runs one deliberately
wrong answer through BOOTH to show that it gets BLOCKED.
"""

import re

try:
    import numpy as np
    import booth

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from sentence_transformers import CrossEncoder
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc}.\n"
        "Install the packages listed in this file's docstring, then "
        "run the script again."
    ) from exc


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "google/flan-t5-base"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
EVIDENCE_THRESHOLD = 0.70

# Small sample knowledge base; no extra data file is needed.
KNOWLEDGE_BASE = """
The library is open from Monday to Friday, 9 AM to 5 PM.
The library is closed on Saturdays and Sundays.
Members can borrow up to five books at a time.
Books must be returned within fourteen days.
A member who returns a book late may need to pay a late fee.
"""


def build_vector_store(text):
    """Split the notes and index them using FAISS."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    return FAISS.from_texts(
        chunks,
        embeddings,
        distance_strategy=DistanceStrategy.COSINE,
    )


class FlanGenerator:
    """Generate an answer using FLAN-T5-base."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL)

    def generate(self, context, question):
        prompt = f"""Answer the question using only the context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
        )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        ).strip()


def make_compare_fn():
    """
    Return a comparison function for BOOTH.

    Each answer sentence is checked against the retrieved evidence.
    The final score is the lowest sentence-level entailment score.
    This is an estimate of support, not a guarantee of truth.
    """
    nli_model = CrossEncoder(NLI_MODEL)

    labels = {
        int(index): str(label).lower()
        for index, label in nli_model.model.config.id2label.items()
    }
    entailment_index = next(
        (index for index, label in labels.items() if "entail" in label),
        None,
    )

    if entailment_index is None:
        raise RuntimeError(
            f"Could not identify the entailment label: {labels}"
        )

    def compare_fn(answer, evidence):
        if not isinstance(answer, str) or not answer.strip():
            return 0.0

        passages = [
            item.strip()
            for item in evidence
            if isinstance(item, str) and item.strip()
        ]
        if not passages:
            return 0.0

        claims = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", answer.strip())
            if sentence.strip()
        ]
        if not claims:
            return 0.0

        # NLI takes (premise, hypothesis): evidence supports the claim.
        pairs = [
            (passage, claim)
            for claim in claims
            for passage in passages
        ]

        logits = np.asarray(nli_model.predict(pairs))
        logits = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        entailment_scores = probabilities[:, entailment_index].reshape(
            len(claims), len(passages)
        )

        # Each claim must have support from at least one retrieved passage.
        best_score_per_claim = entailment_scores.max(axis=1)
        return float(best_score_per_claim.min())

    return compare_fn


def run_wrong_answer_demo(vector_store, compare_fn):
    """Check one deliberately wrong and one correct hardcoded answer."""
    question = "What time does the library close on weekdays?"

    documents = vector_store.similarity_search(question, k=3)
    evidence = [doc.page_content for doc in documents]

    answers = [
        ("wrong (hardcoded)", "The library closes at 9 PM on weekdays."),
        ("correct (hardcoded)", "The library closes at 5 PM on weekdays."),
    ]

    print("\n=== Hardcoded answer demo ===")
    print("Question:", question)

    for label, answer in answers:
        result = booth.check_with_evidence(
            answer=answer,
            evidence=evidence,
            compare_fn=compare_fn,
            evidence_threshold=EVIDENCE_THRESHOLD,
        )
        print(f"\n[{label}] {answer}")
        print("Status:", result.status)
        print("Evidence agreement:", result.evidence_agreement)
        print("Detail:", getattr(result, "detail", None))


def main():
    vector_store = build_vector_store(KNOWLEDGE_BASE)
    generator = FlanGenerator()
    compare_fn = make_compare_fn()

    print("Local RAG + BOOTH evidence check")
    print("Ask about the sample library notes.")
    question = input("\nQuestion: ").strip()

    if not question:
        print("Please enter a question.")
        return

    # Retrieve the actual evidence chunks from FAISS.
    documents = vector_store.similarity_search(question, k=3)
    evidence = [doc.page_content for doc in documents]
    context = "\n".join(evidence)

    print("\n--- Retrieved evidence ---")
    for number, passage in enumerate(evidence, start=1):
        print(f"{number}. {passage}")

    answer = generator.generate(context, question)
    print("\n--- FLAN-T5 answer ---")
    print(answer)

    # BOOTH checks the answer against the same chunks used by FLAN-T5.
    result = booth.check_with_evidence(
        answer=answer,
        evidence=evidence,
        compare_fn=compare_fn,
        evidence_threshold=EVIDENCE_THRESHOLD,
    )

    print("\n--- BOOTH result ---")
    print("Status:", result.status)
    print("Evidence agreement:", result.evidence_agreement)
    print("Detail:", getattr(result, "detail", None))

    if result.status == booth.BLOCKED:
        print("Decision: The answer was blocked by the evidence check.")
    else:
        print(
            "Decision: Review the status and evidence agreement before "
            "using the answer."
        )

    # Show that a deliberately wrong answer is blocked.
    run_wrong_answer_demo(vector_store, compare_fn)


if __name__ == "__main__":
    main()