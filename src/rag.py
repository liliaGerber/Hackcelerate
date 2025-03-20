import sqlite3
import json
import numpy as np
from ollama import chat
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = "memories.sqlite"
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # Efficient small model


def summarize_text(previous_summary, new_messages):
    """Generate a concise updated summary based on the previous summary and new messages."""
    new_content = " ".join([msg["text"] for msg in new_messages if msg.get("text")])
    prompt = f"Previous summary: {previous_summary}\nNew messages: {new_content}\nGenerate an updated concise summary:"
    response = chat(model="gemma3:1b", messages=[{"role": "user", "content": prompt}])
    return response.get("message", {}).get("content", previous_summary)  # Fallback to previous summary


def get_embedding(text):
    """Generate an embedding vector for the given text."""
    return EMBEDDING_MODEL.encode(text).tolist()


def store_memories(chat_session_id, chat_history):
    """Store chat messages asynchronously to improve performance."""
    with sqlite3.connect(DB_PATH) as conn:
        print("connected")
        c = conn.cursor()

        # Fetch existing memory IDs
        all_id_rows = c.execute("SELECT id FROM memories").fetchall()
        print("fetched all memories")
        all_ids = {row[0] for row in all_id_rows}
        current_chats = [msg for msg in chat_history if msg["id"] not in all_ids]

        if not current_chats:
            return

        # Fetch the last summary
        last_summary_row = c.execute(
            "SELECT current_summary FROM memories WHERE chat_session_id = ? ORDER BY id DESC LIMIT 1",
            (chat_session_id,),
        ).fetchone()
        print("fetched last summary row")
        last_summary = last_summary_row[0] if last_summary_row else ""

        # Generate new summary
        new_summary = summarize_text(last_summary, current_chats)
        print("generated new summary")

        # Insert new messages into the database
        for message in current_chats:
            print("message")
            text = message.get("text", "").strip()
            if not text:
                continue

            embedding_str = json.dumps(get_embedding(text))
            c.execute(
                "INSERT INTO memories (id, chat_session_id, text, embedding, entity, current_summary) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (message["id"], chat_session_id, text, embedding_str, message["type"], new_summary),
            )

        conn.commit()


def fetch_relevant_memories(chat_session_id, query, top_k=5):
    """Retrieve the most relevant memories based on a query using cosine similarity."""
    query_embedding = get_embedding(query)

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()

        # Fetch all stored embeddings for the session
        c.execute("SELECT id, text, embedding FROM memories WHERE chat_session_id = ?", (chat_session_id,))
        memories = c.fetchall()

    if not memories:
        return []

    # Compute similarity scores
    memory_texts = []
    memory_embeddings = []
    memory_ids = []

    for mem_id, text, embedding_json in memories:
        embedding = json.loads(embedding_json)
        memory_ids.append(mem_id)
        memory_texts.append(text)
        memory_embeddings.append(embedding)

    memory_embeddings = np.array(memory_embeddings)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    similarities = cosine_similarity(query_embedding, memory_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]  # Get top-k highest scores

    return [{"id": memory_ids[i], "text": memory_texts[i], "score": similarities[i]} for i in top_indices]


def generate_response(query, chat_session_id):
    """Use retrieved memories to generate a response with RAG."""
    relevant_memories = fetch_relevant_memories(chat_session_id, query)

    context = "\n".join([mem["text"] for mem in relevant_memories])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    # return generate_answer_with_llm(prompt)  # Replace with actual LLM function


store_memories("chat_session_id", [
    {'id': 'cdb4733c-bea0-408f-9429-3f4ffa0bfd9c', 'inResponseTo': None,
     'text': 'Hey! I am Hemy, your waiter, what can I do for you?', 'replayId': 'eaed0f39-fc4f-40e3-a052-15319c69769b',
     'isProgressing': False, 'type': 1},
    {'id': 'cfb4733c-bea0-408f-9429-3f4ffa0bfd9c', 'inResponseTo': None,
     'text': 'Can I have a glass of water?', 'replayId': 'eaed0f39-fc4f-40e3-a052-15319c69769b',
     'isProgressing': False, 'type': 0},
    {'id': 'ccb4733c-bea0-408f-9429-3f4ffa0bfd9c', 'inResponseTo': None,
     'text': 'Sure', 'replayId': 'eaed0f39-fc4f-40e3-a052-15319c69769b',
     'isProgressing': False, 'type': 1},
    {'id': 'cfb4733c-bea0-408f-9429-3f4ffa0bfd9c', 'inResponseTo': None,
     'text': 'I really like steak', 'replayId': 'eaed0f39-fc4f-40e3-a052-15319c69769b',
     'isProgressing': False, 'type': 0},
])

print(
    fetch_relevant_memories("chat_session_id", "what's their favorite food?")
)
