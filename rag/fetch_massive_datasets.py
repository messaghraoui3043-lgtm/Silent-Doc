import os
import sys
from pathlib import Path

# Try importing datasets, instruct user to install if missing
try:
    from datasets import load_dataset
except ImportError:
    print("\n[Error] The 'datasets' library is missing!")
    print("Please run: pip install datasets pandas")
    print("Or install the updated requirements.txt\n")
    sys.exit(1)

# ============================================================
# Configuration
# ============================================================
DATASETS = [
    "lonnieqin/skin_cancer_qa",
    "Lili99/Clinical_Notes_Dermatology",
    "medalpaca/medical_meadow_wikidoc",
    "BIOMEDical-NLP/PubMed-General"
]
SPLIT = "train"        # Use the appropriate split ("train", "test", etc.)
BATCH_SIZE = 5000      # Increased batch size for massive knowledge bases
MAX_RECORDS_PER_DATASET = 12500 # Pull 12,500 records per dataset to hit 50k total

def fetch_and_save_data():
    root_dir = Path(__file__).resolve().parent.parent
    db_dir = root_dir / "rag" / "medical_database"
    
    # Ensure our drop folder exists
    os.makedirs(db_dir, exist_ok=True)
    
    print("\n[Database Fetch] Initiating MASSIVE data gathering pipeline...\n")
    
    for dataset_name in DATASETS:
        print(f"[Fetch] Starting download for: '{dataset_name}'")
        try:
            # We strictly request streaming=True on gigantic datasets to prevent RAM exhaustion
            # but standard load_dataset caches the subsets. Safe to load.
            dataset = load_dataset(dataset_name, split=SPLIT)
        except Exception as e:
            print(f"[Error] Failed to load '{dataset_name}': {e}\n  -> Skipping to next dataset...\n")
            continue
            
        total_records = len(dataset)
        capped_records = min(total_records, MAX_RECORDS_PER_DATASET)
        print(f"  -> Successfully downloaded {total_records} records (Capping to {capped_records} to prevent overload).")
        print(f"  -> Saving into {BATCH_SIZE}-record batches...\n")

        batch_idx = 1
        current_batch_content = ""
        records_in_batch = 0
        extracted_total = 0
        safe_name = dataset_name.replace("/", "_")

        for idx, record in enumerate(dataset):
            if extracted_total >= capped_records:
                break
                
            # Extract fields dynamically based on common schemas (Including scientific paper nodes)
            instruction = record.get("instruction", record.get("question", record.get("title", ""))).strip()
            input_text = record.get("input", record.get("context", record.get("contents", ""))).strip()
            output = record.get("output", record.get("answer", record.get("text", record.get("abstract", "")))).strip()
            
            # Format the record logically
            if not output and not instruction:
                continue
                
            md_entry = f"## [Source: {dataset_name}] Medical Record {idx + 1}\n"
            if instruction or input_text:
                question = f"{instruction} {input_text}".strip()
                if question:
                    md_entry += f"**Context/Title/Query:** {question}\n"
            md_entry += f"**Medical Data:** {output}\n\n---\n\n"
            
            current_batch_content += md_entry
            records_in_batch += 1
            extracted_total += 1
            
            # When batch size is reached or we hit our cap, flush to file
            if records_in_batch >= BATCH_SIZE or extracted_total == capped_records or idx == total_records - 1:
                file_name = db_dir / f"{safe_name}_batch_{batch_idx}.md"
                with open(file_name, "w", encoding="utf-8") as f:
                    f.write(current_batch_content)
                    
                print(f"    Saved batch {batch_idx} ({records_in_batch} records) -> {file_name.name}")
                
                # Reset
                batch_idx += 1
                current_batch_content = ""
                records_in_batch = 0
                
        print(f"\n[Fetch] Finished processing '{dataset_name}'. Extracted {extracted_total} target records.\n")

    print("\n[Database Fetch] ALL DATASETS PROCESSED! Your massive RAG drop folder is ready for ingestion.\n")

if __name__ == "__main__":
    fetch_and_save_data()
