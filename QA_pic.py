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
MODEL = "gemma4:31b-cloud"  # image checking in cloud model
OLLAMA_URL = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT = 600
TEMPERATURE = 0.7
SEED = 42   # USESEED below must be True for this seed to take effect
USESEED = True

PICTUREFILE = "Graph.png"
DESCRIPTION_FILE = "QA_image_description.md"
GRAPH_FILE_TEMPLATE = "QA_knowledge_graph_run{}.json"
OUTPUT_FILE = "QA_pic_results.json"

#Second image
PICTUREFILE2 = "Graph2.png"
DESCRIPTION_FILE2 = "QA_image2_description.md"
GRAPH_FILE_TEMPLATE2 = "QA_knowledge_graph2_run{}.json"
OUTPUT_FILE2 = "QA_pic2_results.json"

REPEATS = 5

# When True, an existing image description file is re-used instead of being generated again.
REUSE_IMAGE_DESCRIPTION = True

# When True, existing per-run JSON graph files are re-used instead of being generated again.
REUSE_GRAPH_FILES = True

# Thresholds are chosen empirically and should be calibrated against known-good runs.
# A consistency score below 0.6 indicates that the model varies its output to such an extent that stability is no longer sufficient.
CONSISTENCY_THRESHOLD = 0.60
# Each sentence is compared with its best-matching chunk of the image description. A sentence below this threshold is likely not supported by the image, i.e. a hallucination risk.
SENTENCE_ALIGNMENT_THRESHOLD = 0.5
FAST_CHUNK_TEXT_MODE_SENTENCE_ALIGNMENT_THRESHOLD = 0.186
# ----------------------------------------

# Per test: the layers (name -> (sentence label, prompt hint)) and example relationships for edges.
TESTS = [
    {
        "name": "Image 1",
        "picture_file": PICTUREFILE,
        "description_file": DESCRIPTION_FILE,
        "graph_file_template": GRAPH_FILE_TEMPLATE,
        "output_file": OUTPUT_FILE,
        "layers": {
            "information_layer": ("in Information Layer", "information objects, data, documents."),
            "system_layer": ("in System Layer", "systems."),
            "organization_layer": ("in Organization Layer", "organizations, units, roles."),
            "process_layer": ("in Process Layer", "processes, activities."),
        },
        "edge_examples": "'uses', 'delivers to', 'is owned by'",
    },
    {
        "name": "Image 2",
        "picture_file": PICTUREFILE2,
        "description_file": DESCRIPTION_FILE2,
        "graph_file_template": GRAPH_FILE_TEMPLATE2,
        "output_file": OUTPUT_FILE2,
        "layers": {
            "conceptual_layer": (
                "in Conceptual Layer",
                "ontology categories and rules (e.g. product theory, structure, material type, packaging, fastening element, function/form).",
            ),
            "operational_layer": (
                "in Operational Layer",
                "concrete data instances of the product (e.g. a specific product, flat pack, parts, materials, hardware with article numbers).",
            ),
            "application_layer": (
                "in Application Layer",
                "business use cases: inputs/triggers, graph queries, matching/tracking/analysis functions and their results (e.g. user support, sustainability tracking, delivery optimization).",
            ),
        },
        "edge_examples": "'has main component', 'consists of', 'is linked to', 'contains', 'graph query'",
    },
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
# Maximum number of words in one description chunk (the embedder truncates long input).
MAX_CHUNK_WORDS = 200

FAST_CHUNK_TEXT_MODE = True
if FAST_CHUNK_TEXT_MODE:
    SENTENCE_ALIGNMENT_THRESHOLD = FAST_CHUNK_TEXT_MODE_SENTENCE_ALIGNMENT_THRESHOLD

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
        "options": {"temperature": TEMPERATURE, "seed": SEED if USESEED else None},
    }
    if json_mode:
        # Makes Ollama constrain the output to valid JSON.
        payload["format"] = "json"
    response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()["response"].strip()


def build_description_prompt():
    return "Describe everything in this image in plain text."


def build_json_template(layers):
    return {
        "knowledge_graph": {
            "nodes": {layer: [] for layer in layers},
            "edges": [{"source": "", "relationship": "", "target": ""}],
        }
    }


def build_graph_prompt(test):
    layer_rules = "\n".join(
        f"- {layer}: {hint}" for layer, (_, hint) in test["layers"].items()
    )
    prompt = textwrap.dedent("""
        Analyze the image and fill in the JSON structure below with what the image shows.
        Rules:
        - Each layer in 'nodes' is a list of short strings, one string per element in the image that belongs to that layer.
        {layer_rules}
        - Each edge in 'edges' describes one connection in the image: 'source' and 'target' must be names that exist in 'nodes', and 'relationship' is a short verb phrase (for example {edge_examples}).
        - Only include what is actually visible in the image. Leave a layer as an empty list if nothing belongs to it.
        - Answer with JSON only, no explanations and no code fences.

        JSON structure to fill in:
    """).strip()
    prompt = prompt.format(layer_rules=layer_rules, edge_examples=test["edge_examples"])
    return f"{prompt}\n{json.dumps(build_json_template(test['layers']), indent=2)}\n"





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


def validate_structure(data, layers):
    """Simple structural test. Returns a list of problems (empty list = OK)."""
    problems = []

    if not isinstance(data, dict) or not isinstance(data.get("knowledge_graph"), dict):
        return ["Top level must be an object with the key 'knowledge_graph'."]
    graph = data["knowledge_graph"]

    nodes = graph.get("nodes")
    if not isinstance(nodes, dict):
        problems.append("'nodes' is missing or not an object.")
        nodes = {}
    for layer in layers:
        if layer not in nodes:
            problems.append(f"Layer '{layer}' is missing.")
        elif not isinstance(nodes[layer], list):
            problems.append(f"Layer '{layer}' is not a list.")
        else:
            for i, n in enumerate(nodes[layer]):
                if not node_name(n):
                    problems.append(f"Layer '{layer}' item {i} has no usable name.")
    for layer in nodes:
        if layer not in layers:
            problems.append(f"Unexpected layer '{layer}'.")

    edges = graph.get("edges")
    if not isinstance(edges, list):
        problems.append("'edges' is missing or not a list.")
        edges = []

    all_names = {
        node_name(n)
        for layer in layers
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

    if not any(isinstance(nodes.get(layer), list) and nodes[layer] for layer in layers) and not edges:
        problems.append("The graph is empty (no nodes and no edges).")

    return problems


# ---------------- JSON -> sentences ----------------
def graph_to_sentences(data, layers):
    graph = data["knowledge_graph"]
    sentences = []

    for layer, (label, _) in layers.items():
        for node in graph["nodes"].get(layer, []):
            name = node_name(node)
            if name:
                sentences.append(f"{name} is {label}.")

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
    return SequenceMatcher(None, a_n, b_n, autojunk=False).ratio() # The autojunk false is needed to have strings over 200 characters compared correctly.


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


def classify_risk(consistency_score, low_sentence_count):
    # Lack of support in the image for even a single sentence means hallucination risk.
    if low_sentence_count > 0:
        return "High"
    # Only the consistency check can give Medium.
    if consistency_score < CONSISTENCY_THRESHOLD:
        return "Medium"
    return "Low"


# ---------------- Embeddings & similarity ----------------
# Cosine similarity between two embedding vectors.
def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def chunk_text_word_limited(text, max_words=MAX_CHUNK_WORDS):
    """
    Splits the text by Markdown headings (###), then limits each chunk's
    total word count to max_words. Heading labels are preserved on each chunk.
    Removes Markdown syntax (### and **) and places the clean heading
    at the very top of each chunk without any prefixes.
    """
    if max_words <= 0:
        raise ValueError("max_words must be greater than 0")

    # 1. Split by the heading markers (###)
    sections = re.split(r'(?=###)', text)
    sections = [section.strip() for section in sections if section.strip()]
    chunks = []

    def append_chunks(header, content):
        words = content.split()
        header_word_count = len(header.split())
        if header_word_count > max_words or (
            words and header_word_count == max_words
        ):
            raise ValueError(
                "max_words must allow the heading and at least one body word"
            )
        if not words:
            chunks.append(header)
            return

        body_words_per_chunk = max_words - header_word_count
        for start in range(0, len(words), body_words_per_chunk):
            chunk_content = " ".join(words[start:start + body_words_per_chunk])
            chunks.append(f"{header}\n{chunk_content}")

    # Process the first chunk (the introduction before the first heading)
    if not sections[0].startswith('###'):
        intro_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', sections[0])  # Strip bold syntax (**)
        append_chunks("Introduction", intro_text)
        start_idx = 1
    else:
        start_idx = 0

    # Process the rest of the sections
    for i in range(start_idx, len(sections)):
        current_section = sections[i]

        # 2. Extract the heading text
        header_match = re.match(r'###\s*(.*)', current_section)
        header_title = header_match.group(1).split('\n')[0] if header_match else "Unknown section"

        # Strip bold syntax from the heading title if any exists
        header_title = re.sub(r'\*\*([^*]+)\*\*', r'\1', header_title).strip()

        # 3. Remove the original heading line (### ...) from the content body
        content_without_header = re.sub(r'^###.*?\n', '', current_section).strip()

        # 4. Strip bold syntax (**) from the remaining content body
        clean_content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content_without_header)
        # 5. Create the clean chunk with the heading line at the very top
        append_chunks(header_title, clean_content)

    return chunks

def chunk_text_per_line(text, max_words=MAX_CHUNK_WORDS):
    """
    Splits text by Markdown headings (###).
    Then creates a chunk for each individual line in a section,
    adding the section title at the top of each chunk.
    Removes Markdown syntax (### and **).
    """
    # 1. Split at heading markers (###)
    sections = re.split(r'(?=###)', text)
    sections = [s.strip() for s in sections if s.strip()]

    chunks = []

    def process_content_lines(header, content):
        lines = content.split('\n')
        for line in lines:
            clean_line = line.strip()
            if clean_line:  # Skip empty lines
                chunks.append(f"{header}\n{clean_line}")

    # Process the introduction (before the first heading) - Check the first element [0]
    if sections and not sections[0].startswith('###'):
        intro_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', sections[0])  # Remove bold (**)
        process_content_lines("Introduction", intro_text)
        start_idx = 1
    else:
        start_idx = 0

    # Process the rest of the sections
    for i in range(start_idx, len(sections)):
        current_section = sections[i]

        # 2. Extract the heading text
        header_match = re.match(r'###\s*(.*)', current_section)
        header_title = header_match.group(1).split('\n')[0] if header_match else "Unknown section"
        
        # Strip bold syntax from the heading title
        header_title = re.sub(r'\*\*([^*]+)\*\*', r'\1', header_title).strip()

        # 3. Remove the original heading line (### ...) from the content body
        content_without_header = re.sub(r'^###.*?\n', '', current_section).strip()
        
        # 4. Strip bold syntax (**) from the remaining content body
        clean_content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content_without_header)
        
        # 5. Create chunks based on lines
        process_content_lines(header_title, clean_content)

    return chunks

def chunk_text(text, max_words=MAX_CHUNK_WORDS):
    if FAST_CHUNK_TEXT_MODE:
        return chunk_text_word_limited(text, max_words)
    return chunk_text_per_line(text, max_words)


def check_alignment(sentences, image_text):
    chunks = chunk_text(image_text)

    emb_chunks = embedder.encode(chunks, normalize_embeddings=True)
    emb_sentences = embedder.encode(sentences, normalize_embeddings=True)

    matrix_sentences = np.array(emb_sentences)
    matrix_chunks = np.array(emb_chunks)

    # Compute similarity for all pairs using a matrix dot product.
    # We transpose the chunks matrix (.T) from (num_chunks, embedding_dim) 
    # to (embedding_dim, num_chunks) so the inner dimensions match for multiplication. 
    # This expects normalized embeddings.
    similarity_matrix = np.dot(matrix_sentences, matrix_chunks.T)

    max_cosines = np.max(similarity_matrix, axis=1)

    per_sentence = [
        {"sentence": s, "cosine": float(score)}
        for s, score in zip(sentences, max_cosines)
    ]

    sorted_aligment_per_sentence = sorted(per_sentence, key=lambda x: x["cosine"])

    mean_alignment = sum(p["cosine"] for p in per_sentence) / len(per_sentence)

    return sorted_aligment_per_sentence, mean_alignment




def run_test(test):
    picture_file = test["picture_file"]
    description_file = test["description_file"]
    layers = test["layers"]
    print("\n" + "=" * 60)
    print(f"TEST: {test['name']} ({picture_file})")
    print("=" * 60)

    if not os.path.exists(picture_file):
        print(f"Picture file '{picture_file}' not found. Skipping test.")
        return

    print(f"Using model: {MODEL}, repeats: {REPEATS}")
    print("-" * 60)

    image_b64 = encode_image(picture_file)
    print(f"Read and encoded image from file: {picture_file}")

    # 1. Plain-text description of the image (the reference text). Reused if it already exists.
    print("Describing the image...")
    time.sleep(1)
    if REUSE_IMAGE_DESCRIPTION and os.path.exists(description_file):
        with open(description_file, "r", encoding="utf-8") as f:
            image_text = f.read().strip()
    else:
        print("Asking the model to describe the image...")
        image_text = ask_model(build_description_prompt(), image_b64)
        with open(description_file, "w", encoding="utf-8") as f:
            f.write(image_text)
    print("\n--- Image description ---")
    print(image_text)

    # 2. Repeated runs: knowledge graph as JSON -> structure test -> sentences -> alignment.
    runs = []
    for i in range(1, REPEATS + 1):
        print(f"\n=== Run {i}/{REPEATS} ===")
        graph_file = test["graph_file_template"].format(i)
        data = None
        if REUSE_GRAPH_FILES and os.path.exists(graph_file):
            try:
                with open(graph_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"Knowledge graph: {graph_file}")
            except json.JSONDecodeError as e:
                print(f"Existing {graph_file} is not valid JSON ({e}); regenerating.")

        if data is None:
            raw_graph = ask_model(build_graph_prompt(test), image_b64, json_mode=True)
            try:
                data = extract_json(raw_graph)
            except json.JSONDecodeError as e:
                print(f"Could not parse the model output as JSON: {e}")
                print(raw_graph)
                continue

            with open(graph_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Saved knowledge graph to {graph_file}")

        problems = validate_structure(data, layers)
        if problems:
            print("Structure test FAILED:")
            for p in problems:
                print(f"- {p}")
            continue
        print("Structure test OK")

        # Decode the JSON file and translate it to sentences.
        with open(graph_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        sentences = graph_to_sentences(data, layers)
        if not sentences:
            print("No sentences could be generated from the graph.")
            continue

        per_sentence, mean_alignment = check_alignment(sentences, image_text)
        runs.append({
            "run": i,
            "graph_file": graph_file,
            "sentences": per_sentence,
            "mean_alignment": mean_alignment,
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

    # Sentences are sorted so that the order the model lists things in does not affect consistency.
    consistency = compute_consistency([" ".join(sorted(s["sentence"] for s in r["sentences"])) for r in runs])
    all_cosines = [s["cosine"] for r in runs for s in r["sentences"]]
    low_count = sum(1 for c in all_cosines if c < SENTENCE_ALIGNMENT_THRESHOLD)
    risk = classify_risk(consistency, low_count)

    print("\n--- Metrics ---")
    print(f"Valid runs: {len(runs)}/{REPEATS}")
    print(f"Consistency: {consistency * 100:.2f}%" + ("  (only one valid run, not meaningful)" if len(runs) < 2 else ""))
    print(f"Sentences with low semantic alignment: {low_count}/{len(all_cosines)}")
    print(f"Risk classification: {risk}")

    out_summary = {
        "model": MODEL,
        "repeats": REPEATS,
        "valid_runs": len(runs),
        "image_text": image_text,
        "consistency": consistency,
        "low_sentence_count": low_count,
        "risk": risk,
        "thresholds": {
            "consistency": CONSISTENCY_THRESHOLD,
            "sentence_alignment": SENTENCE_ALIGNMENT_THRESHOLD,
        },
        "runs": runs,
    }
    with open(test["output_file"], "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {test['output_file']}")


def main():
    for test in TESTS:
        input("Press Enter to continue... ")
        run_test(test)



if __name__ == "__main__":
    main()
