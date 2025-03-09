import os
from flask import Flask, request, jsonify
from groq import Groq
import requests
import json
from dataclasses import dataclass, asdict, field
from graphrag import (
    extract_graph_components, ingest_to_neo4j, ingest_to_qdrant, retriever_search,
    neo4j_driver, qdrant_client, fetch_related_graph, format_graph_context, graphRAG_run
)

app = Flask(__name__)

FREEPIK_API_KEY = os.environ.get("FREEPIK_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

journal_store = {}
prompt_store = []

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@dataclass
class JournalDocument:
    title: str = None
    content: str = None
    prompt: str = None
    date: str = None
    attachments: list[dict[str, str]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=4)

    @staticmethod
    def from_json(json_str: str) -> 'JournalDocument':
        return JournalDocument(**json.loads(json_str))

    @staticmethod
    def from_dict(data: dict) -> 'JournalDocument':
        return JournalDocument(**data)

    def to_dict(self) -> dict:
        return asdict(self)

@app.route('/submit_journal', methods=['POST'])
def submit_journal():
    try:
        json_data = request.form.get("metadata")
        if not json_data:
            return jsonify({"error": "Missing JSON metadata"}), 400

        data = json.loads(json_data)
        uploaded_files = []

        if 'attachments' in request.files:
            files = request.files.getlist('attachments')
            for file in files:
                if file.filename:
                    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                    file.save(file_path)
                    uploaded_files.append({
                        "file_name": file.filename,
                        "file_path": file_path,
                        "file_type": file.content_type
                    })

        data["attachments"] = uploaded_files
        journal = JournalDocument.from_dict(data)

        nodes, relationships = extract_graph_components(journal.content)
        node_id_mapping = ingest_to_neo4j(nodes, relationships)

        print(node_id_mapping)
        print(journal.content)
        collection_name = "journal_embeddings"
        ingest_to_qdrant(collection_name, journal.content, node_id_mapping)
        return jsonify({"message": "Journal submitted successfully!", "data": journal.to_dict()}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/journal/prompt", methods=["POST"])
def upload_prompt():
    data = request.json
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400
    prompt_store.append(prompt)
    return jsonify({"message": "Prompt uploaded successfully"})

@app.route("/journal/<title>", methods=["GET"])
def get_journal(title):
    doc = journal_store.get(title)
    if doc:
        return jsonify(json.loads(doc))
    return jsonify({"error": "Journal not found"}), 404

@app.route("/image", methods=["POST"])
def generate_image():
    data = request.json
    prompt = data.get("prompt")
    aspect_ratio = data.get("aspect_ratio", "widescreen_16_9")

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    url = "https://api.freepik.com/v1/ai/mystic"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-freepik-api-key": FREEPIK_API_KEY
    }
    payload = {"prompt": prompt, "aspect_ratio": aspect_ratio}
    response = requests.post(url, json=payload, headers=headers)
    return jsonify(response.json())

@app.route('/retrieve_journal', methods=['POST'])
def retrieve_journal():
    try:
        data = request.json
        query = data.get("query")
        if not query:
            return jsonify({"error": "Missing query"}), 400

        collection_name = "journal_embeddings"
        retriever_result = retriever_search(neo4j_driver, qdrant_client, collection_name, query)

        entity_ids = [item.content.split("'id': '")[1].split("'")[0] for item in retriever_result.items]
        subgraph = fetch_related_graph(neo4j_driver, entity_ids)
        graph_context = format_graph_context(subgraph)
        answer = graphRAG_run(graph_context, query)

        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/journal", methods=["GET"])
def get_journals():
    if not journal_store:
        return jsonify({"message": "No journals found"}), 404
    return jsonify({"journals": list(journal_store.values())}), 200

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])
    if not messages or not isinstance(messages, list):
        return jsonify({"error": "Invalid messages format"}), 400

    response = ""
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        response += content

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
