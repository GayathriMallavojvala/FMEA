"""
Faithfulness Layer — claim-level grounding verification for RAG-generated CAPA reports.

WHY THIS EXISTS
----------------
Standard RAG shows a user retrieved context and a generated answer, and trusts the
model actually used the context faithfully. It usually did not check. For a
compliance-adjacent tool (an auditor will eventually read this output), "trust me"
is not good enough — every claim in a CAPA report should be traceable to *something*,
either the retrieved evidence or explicit, flagged inference.

This module adds a claim-level verification pass on top of the existing RAG pipeline:
  1. Split the generated CAPA text into individual claims (roughly: sentences).
  2. Score each claim's lexical groundedness against the retrieved context using the
     SAME TF-IDF vector space already built for retrieval (cosine similarity) — this
     is instant, deterministic, and needs no extra API call.
  3. Optionally, ask the LLM itself to act as a judge: given the claim and the
     retrieved context, classify it as Grounded / Inferred / Unsupported with a one
     -line reason. This is slower (one API call per scored claim) but catches paraphrases the
     lexical check misses (e.g. "40C overshoot" vs "temperature exceeded setpoint").
  4. Produce an annotated version of the report (inline tags) plus a summary
     faithfulness score.

WHAT THIS DOES NOT CLAIM
--------------------------
This is a heuristic aid, not a formal proof of correctness. The lexical score can
be fooled by paraphrase (hence the optional LLM-judge pass), and the LLM judge can
itself hallucinate its own judgments (hence running both, and being transparent that
they're heuristics with stated thresholds, not ground truth). It is only meaningful
for the RAG arm — the baseline (no-retrieval) arm has no retrieved context to check
claims against, so faithfulness scoring does not apply to it.
"""
import json
import re

# Thresholds for the lexical (TF-IDF cosine similarity) grounding check.
# These are deliberately disclosed and tunable, not presented as precise science.
LEXICAL_GROUNDED_THRESHOLD = 0.22
LEXICAL_INFERRED_THRESHOLD = 0.08

# Claims shorter than this many words are treated as structural (headings, list
# markers) and skipped rather than scored, since scoring "1. Problem Description:"
# against retrieved context is meaningless.
MIN_CLAIM_WORDS = 6

VERIFIER_SYSTEM_PROMPT = """You are a strict fact-checking assistant. You will be given
CONTEXT (source-labeled references and separately reported case evidence) and a CLAIM taken from an AI-generated report.
Classify the claim as exactly one of:
- GROUNDED: the claim's specific facts are directly supported by the retrieved context
- INFERRED: the claim is a reasonable inference/synthesis from the context, but not stated outright
- UNSUPPORTED: the claim's specific facts are not supported by the retrieved context at all

Respond with ONLY a JSON object: {"label": "GROUNDED"|"INFERRED"|"UNSUPPORTED", "reason": "<one short sentence>"}
Grounded means supported by the supplied text, not independently true. Respect source provenance and uncertainty.
No other text."""


def split_into_claims(capa_text):
    """Split generated CAPA text into sentence-ish claims. Naive but effective for
    the fairly formulaic structure this system's system prompt produces."""
    # Keep numbered headings attached to what follows isn't necessary; we only
    # need reasonable sentence boundaries.
    raw = re.split(r"(?<=[.!?:])\s+(?=[A-Z0-9])", capa_text.replace("\n", " "))
    claims = [c.strip() for c in raw if c.strip()]
    return claims


def lexical_ground_score(claim, context, vectorizer):
    """Cosine similarity between a claim and the retrieved context, using the
    SAME fitted TF-IDF vectorizer as retrieval (shared vocabulary => consistent,
    directly comparable scores across claims)."""
    if not context.strip():
        return 0.0
    vecs = vectorizer.transform([claim, context])
    from sklearn.metrics.pairwise import cosine_similarity
    return float(cosine_similarity(vecs[0], vecs[1])[0][0])


def lexical_label(score):
    if score >= LEXICAL_GROUNDED_THRESHOLD:
        return "GROUNDED"
    elif score >= LEXICAL_INFERRED_THRESHOLD:
        return "INFERRED"
    return "UNSUPPORTED"


def annotate_lexical(capa_text, context, vectorizer):
    """Fast, deterministic, no-API-call faithfulness pass. Returns
    (annotated_text, summary_dict)."""
    claims = split_into_claims(capa_text)
    tags = {"GROUNDED": "\u2713", "INFERRED": "~", "UNSUPPORTED": "\u26a0"}
    counts = {"GROUNDED": 0, "INFERRED": 0, "UNSUPPORTED": 0, "skipped": 0}
    annotated_lines = []
    for claim in claims:
        word_count = len(claim.split())
        if word_count < MIN_CLAIM_WORDS:
            annotated_lines.append(claim)
            counts["skipped"] += 1
            continue
        score = lexical_ground_score(claim, context, vectorizer)
        label = lexical_label(score)
        counts[label] += 1
        annotated_lines.append(f"{claim} [{tags[label]} {label.title()} ({score:.2f})]")
    scored_total = sum(counts[k] for k in ("GROUNDED", "INFERRED", "UNSUPPORTED"))
    grounded_rate = (counts["GROUNDED"] / scored_total) if scored_total else 0.0
    summary = {
        "method": "lexical (TF-IDF cosine)",
        "scored_claims": scored_total,
        "grounded": counts["GROUNDED"],
        "inferred": counts["INFERRED"],
        "unsupported": counts["UNSUPPORTED"],
        "grounded_rate": round(grounded_rate, 3),
    }
    return " ".join(annotated_lines), summary


def verify_claims_llm(capa_text, context, groq_client, model):
    """Slower, API-based verification pass: ask the LLM to judge each claim
    against the retrieved context. Returns (annotated_text, summary_dict).
    Raises whatever the Groq client raises on failure — caller should catch."""
    claims = [c for c in split_into_claims(capa_text) if len(c.split()) >= MIN_CLAIM_WORDS]
    results = []
    for claim in claims:
        user_msg = f"RETRIEVED CONTEXT:\n{context}\n\nCLAIM:\n{claim}"
        resp = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        raw = resp.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
            label = parsed.get("label", "UNSUPPORTED").upper()
            reason = parsed.get("reason", "")
        except (json.JSONDecodeError, AttributeError):
            label, reason = "UNSUPPORTED", "(verifier response could not be parsed)"
        if label not in ("GROUNDED", "INFERRED", "UNSUPPORTED"):
            label = "UNSUPPORTED"
        results.append((claim, label, reason))

    tags = {"GROUNDED": "\u2713", "INFERRED": "~", "UNSUPPORTED": "\u26a0"}
    annotated = " ".join(f"{c} [{tags[l]} {l.title()}: {r}]" for c, l, r in results)
    counts = {"GROUNDED": 0, "INFERRED": 0, "UNSUPPORTED": 0}
    for _, l, _ in results:
        counts[l] += 1
    total = len(results)
    grounded_rate = (counts["GROUNDED"] / total) if total else 0.0
    summary = {
        "method": "LLM-as-judge",
        "scored_claims": total,
        "grounded": counts["GROUNDED"],
        "inferred": counts["INFERRED"],
        "unsupported": counts["UNSUPPORTED"],
        "grounded_rate": round(grounded_rate, 3),
    }
    return annotated, summary


def format_summary(summary):
    return (
        f"Faithfulness check ({summary['method']}): "
        f"{summary['grounded']}/{summary['scored_claims']} claims grounded "
        f"({summary['grounded_rate']*100:.0f}%), "
        f"{summary['inferred']} inferred, {summary['unsupported']} unsupported."
    )
