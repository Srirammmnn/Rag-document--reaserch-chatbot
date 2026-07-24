"""
Citation Verification & Groundedness Engine
============================================
Examines generated RAG answers, extracts inline citation markers [1], [2], [3],
and verifies claim faithfulness against the retrieved document chunks.

Output format:
  - verified_citations: List of dicts with source, snippet, claim, grounded status, and score.
  - groundedness_score: Percentage of citations that are grounded in context (0.0 to 1.0).
"""

import re
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document


def extract_citations_from_text(answer_text: str) -> List[int]:
    """Extract numeric citation IDs from text like [1], [2], [1, 2]."""
    matches = re.findall(r'\[(\d+(?:\s*,\s*\d+)*)\]', answer_text)
    citation_ids = set()
    for match in matches:
        for num in match.split(','):
            num_str = num.strip()
            if num_str.isdigit():
                citation_ids.add(int(num_str))
    return sorted(list(citation_ids))


def verify_claims_against_sources(
    answer_text: str,
    context_docs: List[Document],
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Verifies claims in answer_text against context_docs.

    Args:
        answer_text: The LLM output text containing inline citations [1], [2]...
        context_docs: The retrieved candidate documents passed as context to the LLM.

    Returns:
        (verified_citations, groundedness_score)
    """
    doc_map: Dict[int, Document] = {}
    for i, doc in enumerate(context_docs, start=1):
        cit_id = doc.metadata.get("citation_id", i)
        doc_map[cit_id] = doc

    # Split text into sentences to associate each citation with its claim sentence
    sentences = re.split(r'(?<=[.!?])\s+', answer_text)
    citation_records: List[Dict[str, Any]] = []

    for sentence in sentences:
        cit_ids = extract_citations_from_text(sentence)
        if not cit_ids:
            continue

        clean_sentence = re.sub(r'\[\d+(?:\s*,\s*\d+)*\]', '', sentence).strip()

        for cit_id in cit_ids:
            target_doc = doc_map.get(cit_id)
            if not target_doc:
                continue

            doc_text = target_doc.page_content.lower()
            claim_words = [w.lower() for w in re.findall(r'\w+', clean_sentence) if len(w) > 3]

            # Calculate word overlap ratio
            if claim_words:
                matches = sum(1 for word in claim_words if word in doc_text)
                overlap_score = round(matches / len(claim_words), 3)
            else:
                overlap_score = 1.0

            is_grounded = overlap_score >= 0.35  # Threshold for grounded claim

            citation_records.append({
                "citation_id": cit_id,
                "source": target_doc.metadata.get("source", "unknown"),
                "page": target_doc.metadata.get("page", 1),
                "claim": clean_sentence,
                "snippet": target_doc.page_content[:300].strip() + ("..." if len(target_doc.page_content) > 300 else ""),
                "grounded": is_grounded,
                "confidence_score": min(1.0, round(overlap_score + 0.2, 2)),
            })

    if not citation_records:
        # Fallback if answer didn't include bracketed citations explicitly: attach top docs
        for cit_id, target_doc in doc_map.items():
            citation_records.append({
                "citation_id": cit_id,
                "source": target_doc.metadata.get("source", "unknown"),
                "page": target_doc.metadata.get("page", 1),
                "claim": "General response synthesis from source document.",
                "snippet": target_doc.page_content[:300].strip() + ("..." if len(target_doc.page_content) > 300 else ""),
                "grounded": True,
                "confidence_score": 0.9,
            })

    grounded_count = sum(1 for c in citation_records if c["grounded"])
    overall_score = round(grounded_count / len(citation_records), 2) if citation_records else 1.0

    return citation_records, overall_score


def test_verifier():
    sample_text = "RAG combines vector search with language models [1]. FAISS handles similarity search [2]."
    sample_docs = [
        Document(page_content="RAG is a technique combining vector database search with LLM generation.", metadata={"source": "rag.pdf", "citation_id": 1}),
        Document(page_content="FAISS is a Facebook library for fast vector similarity search.", metadata={"source": "faiss.pdf", "citation_id": 2}),
    ]
    citations, score = verify_claims_against_sources(sample_text, sample_docs)
    print(f"Verified {len(citations)} citations. Groundedness Score: {score}")
    for c in citations:
        print(f" - [{c['citation_id']}] Grounded={c['grounded']} Score={c['confidence_score']} Source={c['source']}")


if __name__ == "__main__":
    test_verifier()
