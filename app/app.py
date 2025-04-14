import os
from flask import Flask, request, jsonify, render_template
from groq import Groq
import requests
import json
from dataclasses import dataclass, asdict, field
from graphrag import (
    extract_graph_components, ingest_to_neo4j, ingest_to_qdrant, retriever_search,
    neo4j_driver, qdrant_client, fetch_related_graph, format_graph_context, graphRAG_run
)

POSSIBLE_TREATMENTS = [
    {
        "treatment_name": "Box Breathing",
        "treatment_description": "A breathing technique where you inhale, hold, exhale, and hold again for 4 seconds each",
        "helps_treat": ["Anxiety", "Stress", "Panic"],
        "complexity": "Low"
    },
    {
        "treatment_name": "Gratitude Journaling",
        "treatment_description": "Write down 3 things you're grateful for each day to shift focus from negative thoughts",
        "helps_treat": ["Depression", "Low Self-Esteem", "Anxiety"],
        "complexity": "Low"
    },
    {
        "treatment_name": "Daily Walk",
        "treatment_description": "Walk outside for 15–30 minutes to promote movement, sunlight exposure, and reflection",
        "helps_treat": ["Depression", "Anxiety", "Burnout"],
        "complexity": "Low"
    },
    {
        "treatment_name": "Digital Detox Hour",
        "treatment_description": "Turn off social media and screens for one hour daily to reduce overstimulation",
        "helps_treat": ["Anxiety", "Doomscrolling", "Sleep Problems"],
        "complexity": "Medium"
    },
    {
        "treatment_name": "Mindful Shower",
        "treatment_description": "Use a daily shower as a mindfulness practice—focus on scent, water temperature, and breathing",
        "helps_treat": ["Anxiety", "Dissociation", "Stress"],
        "complexity": "Low"
    },
    {
        "treatment_name": "Name the Emotion",
        "treatment_description": "Label what you're feeling (\"I'm anxious\" or \"I'm overwhelmed\") to regain control",
        "helps_treat": ["Emotional Dysregulation", "Anxiety", "Overthinking"],
        "complexity": "Low"
    },
    {
        "treatment_name": "5-4-3-2-1 Grounding",
        "treatment_description": "Use your senses to find 5 things you can see, 4 touch, 3 hear, 2 smell, 1 taste",
        "helps_treat": ["Panic Attacks", "Anxiety", "Dissociation"],
        "complexity": "Medium"
    },
    {
        "treatment_name": "Routine Anchors",
        "treatment_description": "Add consistent “anchor” activities to your day (e.g., tea at 3pm, bedtime stretches)",
        "helps_treat": ["Depression", "ADHD", "Anxiety"],
        "complexity": "Medium"
    },
    {
        "treatment_name": "Positive Reframing",
        "treatment_description": "Practice catching and reframing negative thoughts into more balanced ones",
        "helps_treat": ["Anxiety", "Depression", "Negative Self-Talk"],
        "complexity": "High"
    },
    {
        "treatment_name": "Social Check-Ins",
        "treatment_description": "Text or call one trusted friend or family member each day, even briefly",
        "helps_treat": ["Loneliness", "Depression", "Low Motivation"],
        "complexity": "Medium"
    },
    {
        "treatment_name": "Progress Tracking",
        "treatment_description": "Track mood or habits using a simple checklist to spot patterns and celebrate small wins",
        "helps_treat": ["Depression", "Anxiety", "Executive Dysfunction"],
        "complexity": "Medium"
    },
    {
        "treatment_name": "Stretch & Breathe Break",
        "treatment_description": "Take a 5-minute stretch and breathing break every few hours during the day",
        "helps_treat": ["Stress", "Anxiety", "Fatigue"],
        "complexity": "Low"
    },
    {
        "treatment_name": "Affirmation Practice",
        "treatment_description": "Repeat positive affirmations like \"I can handle this\" or \"I am safe\" out loud or in writing",
        "helps_treat": ["Anxiety", "Self-Esteem", "Depression"],
        "complexity": "Low"
    },
    {
        "treatment_name": "Worry Time",
        "treatment_description": "Schedule 15 minutes per day just for worrying—outside that window, gently postpone thoughts",
        "helps_treat": ["Generalized Anxiety", "Obsessive Thinking"],
        "complexity": "High"
    },
    {
        "treatment_name": "Sleep Wind-Down Ritual",
        "treatment_description": "Create a consistent, calming bedtime routine: dim lights, no screens, light reading or journaling",
        "helps_treat": ["Insomnia", "Anxiety", "Stress"],
        "complexity": "Medium"
    }
]

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
    tags: str = None
    date: str = None
    attachments: list[dict[str, str]] = field(default_factory=list)
    additionalInfo: str = None

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

@app.route("/index")
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/insights")
def insights():
    return render_template("insights.html")

@app.route("/treatment")
def treatment():
    return render_template("treatment.html")

@app.route("/generate_treatment", methods=["POST"])
def generate_treatment():
    based_on_treatments = True
    if based_on_treatments:
        treatment_type = f"""
            Use the treatments for suggestions:
        
            {POSSIBLE_TREATMENTS}
        """
    else:
         treatment_type ="""
            Give your treatment based on what is best for the user, and not too serious
         """

    prompt = f"""
        You are a helpful assistant, helping someone explore and improve their mental health. They have written journal
        entries in the format of
        
        --- 
        
        Here are the relevant journal entries
        
        --- (graph-rag, result?)
        
        Here are some previous insights generated by the application
        
        ---

        
        Generate a treatment plan with these instructions:
        
        {treatment_type}
    """

    response = requests.post("http://127.0.0.1:5000/retrieve_journal", json={"query": prompt})


    response_data = response.json()

    print("RESPONSE_DATS")
    print(response_data)
    print(type(response_data))
    print(response_data["response"])
    print(type(response_data["response"]))

    return jsonify({"response": response_data["response"]}), response.status_code

@app.route('/submit_journals', methods=['POST'])
def submit_journals():
    try:
        entries = request.json.get("entries")

        if not entries:
            return jsonify({"error": "No journal entries passed"}), 400


        print("Entries: ", entries)
        if not isinstance(entries, list):
            return jsonify({"error": "Entries should be a list of journal entries"}), 400

        responses = []

        for idx, entry in enumerate(entries):
            try:
                # Handle attachments for each entry
                uploaded_files = []
                attachment_key = f'attachments_{idx}'
                if attachment_key in request.files:
                    files = request.files.getlist(attachment_key)
                    for file in files:
                        if file.filename:
                            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                            file.save(file_path)
                            uploaded_files.append({
                                "file_name": file.filename,
                                "file_path": file_path,
                                "file_type": file.content_type
                            })

                # Add the uploaded files to the entry data
                entry["attachments"] = uploaded_files
                print("Entry:", entry)
                # Create a JournalDocument instance
                journal = JournalDocument.from_dict(entry)

                print("Journal: ", journal)

                # Extract graph components
                nodes, relationships = extract_graph_components(journal.content)
                node_id_mapping = ingest_to_neo4j(nodes, relationships)

                print("Got graph relations")

                # Ingest to Qdrant
                collection_name = "journal_embeddings"
                ingest_to_qdrant(collection_name, journal.content, node_id_mapping)

                responses.append({
                    "index": idx,
                    "message": "Journal submitted successfully",
                    "data": journal.to_dict()
                })
            except Exception as e:
                responses.append({
                    "index": idx,
                    "error": str(e)
                })

        return jsonify(responses), 207

    except Exception as e:
        return jsonify({"error": str(e)}), 500





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

        print("Made it")

        print(node_id_mapping)
        print(journal.content)
        collection_name = "journal_embeddings"
        ingest_to_qdrant(collection_name, journal.content, node_id_mapping)

        print("Ingested")
        return jsonify({"message": "Journal submitted successfully!", "data": journal.to_dict()}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

        print(answer)

        return jsonify({"response": answer.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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


@app.route("/journal", methods=["GET"])
def get_journals():
    if not journal_store:
        return jsonify({"message": "No journals found"}), 404
    return jsonify({"journals": list(journal_store.values())}), 200


# @app.route("/journal/images", methods=["POST"])
# def generateJournalImages():
#     journal =

if __name__ == "__main__":
    app.run(debug=True)
