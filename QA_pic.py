print("Starting QA for image analysis")
# QA-project.
"""
Image -> description + knowledge graph (JSON) -> sentences -> embeddings -> cosine similarity.
Uses a vision model via the Ollama HTTP API + sentence-transformers embeddings.
Place Graph.png in same folder as this script.
Requires:
pip install sentence-transformers numpy requests
"""

import base64
import json
import os
import re
import time
from difflib import SequenceMatcher
import textwrap

import numpy as np
import requests
from sentence_transformers import SentenceTransformer


# ---------------- CONFIG ----------------
# MODEL = "llama3.2:1b"
MODEL = "gemma4:31b-cloud"  # image checking in cloud model
OLLAMA_URL = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT = 600

PICTUREFILE = "Graph.png"
DESCRIPTION_FILE = "QA_image_description.txt"
GRAPH_FILE_TEMPLATE = "QA_knowledge_graph_run{}.json"
OUTPUT_FILE = "qa_pic_results.json"
REPEATS = 1

# Thresholds are chosen empirically and should be calibrated against known-good runs.
# A consistency score below 0.6 indicates that the model varies its output to such an extent that stability is no longer sufficient.
CONSISTENCY_THRESHOLD = 0.60
# A single short sentence covers only a fraction of the image text, so its threshold is lower than the overall one. A sentence below this threshold is likely not supported by the image, i.e. a hallucination risk.
SENTENCE_ALIGNMENT_THRESHOLD = 0.20
# The overall alignment threshold indicates how well sentencified JSON-structure aligns with the description of the image.
OVERALL_ALIGNMENT_THRESHOLD = 0.60
# ----------------------------------------

LAYERS = [
    "information_layer",
    "system_layer",
    "organization_layer",
    "process_layer",
]

# Used to phrase a node as a natural sentence.
LAYER_LABELS = {
    "information_layer": "in Information Layer",
    "system_layer": "in System Layer",
    "organization_layer": "in Organization Layer",
    "process_layer": "in Process Layer",
}

JSON_TEMPLATE = {
    "knowledge_graph": {
        "nodes": {layer: [] for layer in LAYERS},
        "edges": [{"source": "", "relationship": "", "target": ""}],
    }
}

embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------- Model calls ----------------
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_model(prompt, image_b64, json_mode=False):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }
    if json_mode:
        # Makes Ollama constrain the output to valid JSON.
        payload["format"] = "json"
    response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()["response"].strip()


def build_description_prompt():
    return "Describe everything in this image in plain text."


def build_graph_prompt():
    prompt_base = textwrap.dedent("""
        Analyze the image and fill in the JSON structure below with what the image shows.
        Rules:
        - Each layer in 'nodes' is a list of short strings, one string per element in the image that belongs to that layer.
        - information_layer: information objects, data, documents.
        - system_layer: systems.
        - organization_layer: organizations, units, roles.
        - process_layer: processes, activities.
        - Each edge in 'edges' describes one connection in the image: 'source' and 'target' must be names that exist in 'nodes', and 'relationship' is a short verb phrase (for example 'uses', 'delivers to', 'is owned by').
        - Only include what is actually visible in the image. Leave a layer as an empty list if nothing belongs to it.
        - Answer with JSON only, no explanations and no code fences.

        JSON structure to fill in:
    """).strip()
    
    return f"{prompt_base}\n{json.dumps(JSON_TEMPLATE, indent=2)}\n"






def extract_json(text):
    # Fallback for models that wrap the JSON in code fences or extra text.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


# ---------------- JSON structure test ----------------
def node_name(node):
    if isinstance(node, str):
        return node.strip()
    if isinstance(node, dict):
        for key in ("name", "label", "id"):
            if isinstance(node.get(key), str):
                return node[key].strip()
    return ""


def validate_structure(data):
    """Simple structural test. Returns a list of problems (empty list = OK)."""
    problems = []

    if not isinstance(data, dict) or not isinstance(data.get("knowledge_graph"), dict):
        return ["Top level must be an object with the key 'knowledge_graph'."]
    graph = data["knowledge_graph"]

    nodes = graph.get("nodes")
    if not isinstance(nodes, dict):
        problems.append("'nodes' is missing or not an object.")
        nodes = {}
    for layer in LAYERS:
        if layer not in nodes:
            problems.append(f"Layer '{layer}' is missing.")
        elif not isinstance(nodes[layer], list):
            problems.append(f"Layer '{layer}' is not a list.")
        else:
            for i, n in enumerate(nodes[layer]):
                if not node_name(n):
                    problems.append(f"Layer '{layer}' item {i} has no usable name.")
    for layer in nodes:
        if layer not in LAYERS:
            problems.append(f"Unexpected layer '{layer}'.")

    edges = graph.get("edges")
    if not isinstance(edges, list):
        problems.append("'edges' is missing or not a list.")
        edges = []

    all_names = {
        node_name(n)
        for layer in LAYERS
        if isinstance(nodes.get(layer), list)
        for n in nodes[layer]
    }
    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            problems.append(f"Edge {i} is not an object.")
            continue
        for key in ("source", "relationship", "target"):
            if not isinstance(edge.get(key), str) or not edge[key].strip():
                problems.append(f"Edge {i} has empty or missing '{key}'.")
        for key in ("source", "target"):
            value = edge.get(key)
            if isinstance(value, str) and value.strip() and value.strip() not in all_names:
                problems.append(f"Edge {i} '{key}' ('{value}') does not exist in nodes.")

    if not any(isinstance(nodes.get(layer), list) and nodes[layer] for layer in LAYERS) and not edges:
        problems.append("The graph is empty (no nodes and no edges).")

    return problems


# ---------------- JSON -> sentences ----------------
def graph_to_sentences(data):
    graph = data["knowledge_graph"]
    sentences = []

    for layer in LAYERS:
        for node in graph["nodes"].get(layer, []):
            name = node_name(node)
            if name:
                sentences.append(f"{name} is {LAYER_LABELS[layer]}.")

    for edge in graph.get("edges", []):
        source = edge.get("source", "").strip()
        relationship = edge.get("relationship", "").strip()
        target = edge.get("target", "").strip()
        if source and relationship and target:
            sentences.append(f"{source} {relationship} {target}.")

    return sentences


# ---------------- Consistency ----------------
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


def compute_consistency(outputs):
    n = len(outputs)
    if n < 2:
        return 1.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += similarity(outputs[i], outputs[j])
            pairs += 1
    return total / pairs if pairs else 0.0


def classify_risk(consistency_score, overall_score, low_sentence_count):
    # Lack of support in the image (overall or for even a single sentence) means hallucination risk.
    if overall_score < OVERALL_ALIGNMENT_THRESHOLD or low_sentence_count > 0:
        return "High"
    # Only the consistency check can give Medium.
    if consistency_score < CONSISTENCY_THRESHOLD:
        return "Medium"
    return "Low"


# ---------------- Embeddings & similarity ----------------
# Cosine similarity between two embedding vectors.
def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def check_alignment(sentences, image_text):
    emb_image = embedder.encode(image_text)

    emb_sentences = embedder.encode(sentences)
    per_sentence = [
        {"sentence": s, "cosine": cosine(e, emb_image)}
        for s, e in zip(sentences, emb_sentences)
    ]

    full_text = " ".join(sentences)
    overall = cosine(embedder.encode(full_text), emb_image)

    return full_text, per_sentence, overall




def main():
    if not os.path.exists(PICTUREFILE):
        print(f"Picture file '{PICTUREFILE}' not found.")
        return

    print(f"Using model: {MODEL}, repeats: {REPEATS}")
    print("-" * 60)

    placeholder = input("Press Enter to continue... ")

    image_b64 = encode_image(PICTUREFILE)
    print(f"Read and encoded image from file: {PICTUREFILE}")

    # 1. Plain-text description of the image (the reference text). Reused if it already exists.
    print("Describing the image...")
    time.sleep(1)
    if os.path.exists(DESCRIPTION_FILE):
        with open(DESCRIPTION_FILE, "r", encoding="utf-8") as f:
            image_text = f.read().strip()
    else:
        print("Asking the model to describe the image...")
        image_text = ask_model(build_description_prompt(), image_b64)
        with open(DESCRIPTION_FILE, "w", encoding="utf-8") as f:
            f.write(image_text)
    print("\n--- Image description ---")
    print(image_text)

    # 2. Repeated runs: knowledge graph as JSON -> structure test -> sentences -> alignment.
    runs = []
    for i in range(1, REPEATS + 1):
        print(f"\n=== Run {i}/{REPEATS} ===")
        raw_graph = ask_model(build_graph_prompt(), image_b64, json_mode=True)
        try:
            data = extract_json(raw_graph)
        except json.JSONDecodeError as e:
            print(f"Could not parse the model output as JSON: {e}")
            print(raw_graph)
            continue

        graph_file = GRAPH_FILE_TEMPLATE.format(i)
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved knowledge graph to {graph_file}")

        problems = validate_structure(data)
        if problems:
            print("Structure test FAILED:")
            for p in problems:
                print(f"- {p}")
            continue
        print("Structure test OK")

        # Decode the JSON file and translate it to sentences.
        with open(graph_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        sentences = graph_to_sentences(data)
        if not sentences:
            print("No sentences could be generated from the graph.")
            continue

        full_text, per_sentence, overall = check_alignment(sentences, image_text)
        runs.append({
            "run": i,
            "graph_file": graph_file,
            "sentences": per_sentence,
            "combined_text": full_text,
            "overall_alignment": overall,
        })
        time.sleep(0.2)

    if not runs:
        print("\nNo valid runs. Nothing to evaluate.")
        return

    # 3. Report per run.
    for r in runs:
        print(f"\n--- Run {r['run']}: sentences and alignment (vs. image text) ---")
        for item in r["sentences"]:
            flag = "" if item["cosine"] >= SENTENCE_ALIGNMENT_THRESHOLD else "  <-- low"
            print(f"{item['cosine']:.3f}  {item['sentence']}{flag}")
        print(f"\nCombined text: {r['combined_text']}")
        print(f"Overall alignment: {r['overall_alignment']:.3f}")

    # Sentences are sorted so that the order the model lists things in does not affect consistency.
    consistency = compute_consistency([" ".join(sorted(s["sentence"] for s in r["sentences"])) for r in runs])
    avg_overall = sum(r["overall_alignment"] for r in runs) / len(runs)
    all_cosines = [s["cosine"] for r in runs for s in r["sentences"]]
    avg_sentence = sum(all_cosines) / len(all_cosines)
    low_count = sum(1 for c in all_cosines if c < SENTENCE_ALIGNMENT_THRESHOLD)
    risk = classify_risk(consistency, avg_overall, low_count)

    print("\n--- Metrics ---")
    print(f"Valid runs: {len(runs)}/{REPEATS}")
    print(f"Consistency: {consistency:.3f}" + ("  (only one valid run, not meaningful)" if len(runs) < 2 else ""))
    print(f"Average sentence alignment: {avg_sentence:.3f}")
    print(f"Sentences below {SENTENCE_ALIGNMENT_THRESHOLD:.2f}: {low_count}/{len(all_cosines)}")
    print(f"Overall alignment (avg): {avg_overall:.3f}")
    print(f"Risk classification: {risk}")

    out_summary = {
        "model": MODEL,
        "repeats": REPEATS,
        "valid_runs": len(runs),
        "image_text": image_text,
        "consistency": consistency,
        "avg_sentence_alignment": avg_sentence,
        "avg_overall_alignment": avg_overall,
        "low_sentence_count": low_count,
        "risk": risk,
        "thresholds": {
            "consistency": CONSISTENCY_THRESHOLD,
            "sentence_alignment": SENTENCE_ALIGNMENT_THRESHOLD,
            "overall_alignment": OVERALL_ALIGNMENT_THRESHOLD,
        },
        "runs": runs,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
