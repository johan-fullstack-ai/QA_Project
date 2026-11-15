# Proof of Concept, QA-project.
print("Starting POC...")
"""
repeated runs + semantic source alignment + heuristics
Uses Ollama Python API + sentence-transformers embeddings.
Python 3.14.6 compatible.
Place jupitermoons.txt in same folder as this script.
Requires:
pip install sentence-transformers numpy ollama
"""

import json
import os
import time
import re
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
import numpy as np
import ollama

# MODEL = "llama3.2:1b" # local model test
MODEL = "gemma4:31b-cloud"
SOURCE_FILE = "jupitermoons.txt"
QUESTION = "How many moons does Jupiter have?"
OUTPUT_FILE = "poc_results_semantic.json"
REPEATS = 5

# A consistency score below 0.6 indicates that the model varies its output to such an extent that stability is no longer sufficient.
CONSISTENCY_THRESHOLD = 0.60
# Threshold 0.6 is chosen empirically because grounded responses consistently end up around 0.63–0.68 in cosine similarity.
SEMANTIC_ALIGNMENT_THRESHOLD = 0.60

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def read_source(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def build_prompt(source_text, QUESTION):
    return (
        "Use the source text to answer the question. "
#        "Answer in one sentence and do not refer to previous responses nor corrections.\n\n"
        "SOURCE:\n"
        f"{source_text}\n\n"
        "Question:\n"
        f"{QUESTION}\n"
    )

def normalize_text(s):
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()

# The method returns a float representing the sequences' similarity. It is calculated as 2M/T, where M is the number of matching elements and T is the total number of elements in both sequences. The result ranges from 0.0 (no similarity) to 1.0 (identical sequences).
def similarity(a, b):
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()

# Takes the source text and the model's response, creates embeddings for both, and calculates cosine similarity.
def semantic_alignment(output, source):
    emb_source = embedder.encode(source)
    emb_output = embedder.encode(output)
    cos = np.dot(emb_source, emb_output) / (np.linalg.norm(emb_source) * np.linalg.norm(emb_output))
    return float(cos)

def compute_consistency(outputs):
    n = len(outputs)
    if n < 2:
        return 1.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i+1, n):
            total += similarity(outputs[i], outputs[j])
            pairs += 1
    return total / pairs if pairs else 0.0

def classify_risk(consistency_score, semantic_score):
    if semantic_score < SEMANTIC_ALIGNMENT_THRESHOLD:
        return "High"
    if consistency_score < CONSISTENCY_THRESHOLD:
        return "Medium"
    return "Low"

def main():
    if not os.path.exists(SOURCE_FILE):
        print(f"Source file '{SOURCE_FILE}' not found.")
        return

    source_text = read_source(SOURCE_FILE)
    prompt = build_prompt(source_text, QUESTION)

    print(f"Model: {MODEL}, repeats: {REPEATS}")
    print("-" * 60)
    placeholder = input("Press Enter to continue... ")

    print("Running PoC with semantic alignment...")
    print(f"Question: {QUESTION}")

    outputs = []

    for i in range(REPEATS):
        print(f"Run {i+1}/{REPEATS} ...", end=" ", flush=True)
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        out = response["message"]["content"].strip()
        outputs.append(out)
        print("done.")
        time.sleep(0.2)

    print("\n--- Outputs ---")
    print(f"Model: {MODEL}")
    for idx, o in enumerate(outputs, 1):
        print(f"[{idx}] {o}\n")

    consistency = compute_consistency(outputs)
    semantic_scores = [semantic_alignment(o, source_text) for o in outputs]
    avg_semantic_score = sum(semantic_scores) / len(semantic_scores)

    risk = classify_risk(consistency, avg_semantic_score)

    print("--- Metrics ---")
    print(f"Consistency: {consistency:.3f}")
    print(f"Semantic alignment: {avg_semantic_score:.3f}")
    print(f"Risk classification: {risk}")

    print("\n--- Per-run semantic alignment ---")
    for i, s in enumerate(semantic_scores, 1):
        print(f"Run {i}: {s:.3f}")

    out_summary = {
        "model": MODEL,
        "repeats": REPEATS,
        "consistency": consistency,
        "avg_semantic_alignment": avg_semantic_score,
        "risk": risk,
        "outputs": outputs,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
