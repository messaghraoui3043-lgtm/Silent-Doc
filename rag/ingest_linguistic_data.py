import os
import chromadb
from pathlib import Path
from datasets import load_dataset
import time

def ingest_linguistic_data():
    root_dir = Path(__file__).resolve().parent.parent
    db_path = root_dir / "rag" / "chroma_db"

    print("[Ingest] Downloading 'Williamsanderson/MedQA-Darija-MultiLingual' from Hugging Face...")
    try:
        # Load the dataset
        dataset = load_dataset("Williamsanderson/MedQA-Darija-MultiLingual", "default", split="train")
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        return

    questions = []
    responses = []
    
    print("[Ingest] Extracting Darija records...")
    # Extract only the columns we need to avoid audio decoding errors
    for item in dataset.select_columns(["question_darija", "answer_darija"]):
        q = item.get("question_darija")
        a = item.get("answer_darija")
        if q and a:
            questions.append(q.strip())
            responses.append(a.strip())

    total_records = len(questions)
    if total_records == 0:
        print("[Error] No valid Darija records found.")
        return

    print(f"[Ingest] Found {total_records} valid linguistic examples.")
    print(f"[Ingest] Initializing ChromaDB Persistent Client at {db_path}...")
    
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name="linguistic_knowledge",
        metadata={"hnsw:space": "cosine"}
    )

    print("[Ingest] Batch embedding & indexing all linguistic chunks into ChromaDB...")
    start_time = time.time()
    
    batch_size = 5000
    try:
        for i in range(0, total_records, batch_size):
            batch_q = questions[i:i + batch_size]
            batch_a = responses[i:i + batch_size]
            batch_ids = [f"ling_{j}" for j in range(i, i + len(batch_q))]
            
            # Store the patient question as the document to embed
            # Store the doctor's response as metadata, so we can retrieve it
            batch_metadata = [{"response": a} for a in batch_a]
            
            print(f"   Processing batch {i} to {i+len(batch_q)} / {total_records}...")
            collection.add(
                documents=batch_q,
                metadatas=batch_metadata,
                ids=batch_ids
            )
            
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return

    print("--------------------------------------------------")
    print(f"INGESTION SUCCESS: Validated {collection.count()} linguistic vectors in ChromaDB.")
    print(f"Time Taken: {round(time.time() - start_time, 2)} seconds")
    print("--------------------------------------------------")

if __name__ == "__main__":
    ingest_linguistic_data()
