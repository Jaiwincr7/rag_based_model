import json
from collections import defaultdict
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
JSON_PATH = "enterprise-attack.json"
CHROMA_PATH = "./chroma_db/mitre_attack_v5"
COLLECTION_NAME = "mitre_enterprise_attack_v5"

def get_external_id(obj):
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None

def main():
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        objects = bundle.get("objects", [])
    except FileNotFoundError:
        print(f"❌ Error: '{JSON_PATH}' not found.")
        return

    # --- 1. Registry with Detection Data ---
    registry = {} 
    
    for obj in objects:
        if obj.get("x_mitre_deprecated") or obj.get("revoked"): continue
        stix_id = obj.get("id")
        mitre_id = get_external_id(obj)
        
        if stix_id and mitre_id:
            registry[stix_id] = {
                "mitre_id": mitre_id,
                "name": obj.get("name"),
                "type": obj.get("type"),
                "description": obj.get("description", ""),
                # Capture Detection logic (usually a paragraph of text)
                "detection": obj.get("x_mitre_detection", "No specific detection logic provided in STIX."),
                "tactics": []
            }

            for phase in obj.get("kill_chain_phases", []):
                if phase.get("kill_chain_name") == "mitre-attack":
                    tactic = phase.get("phase_name").replace("-", " ").lower()
                    registry[stix_id]["tactics"].append(tactic)

    # --- 2. Build Hard Links ---
    tech_mitigated_by = defaultdict(list)
    mitigation_targets = defaultdict(list)

    print("🔗 Building Relationship Graph...")
    for obj in objects:
        if obj.get("type") == "relationship" and obj.get("relationship_type") == "mitigates":
            src = obj.get("source_ref")
            tgt = obj.get("target_ref")
            
            if src in registry and tgt in registry:
                src_meta = registry[src]
                tgt_meta = registry[tgt]
                mitigation_targets[src].append(f"{tgt_meta['mitre_id']} {tgt_meta['name']}")
                tech_mitigated_by[tgt].append(f"{src_meta['mitre_id']} {src_meta['name']}")

    # --- 3. Indexing ---
    documents = []
    print("📝 Indexing Objects...")

    for stix_id, data in registry.items():
        obj_type = data["type"]
        if obj_type not in ["attack-pattern", "course-of-action"]:
            continue

        meta = {
            "mitre_id": data["mitre_id"],
            "name": data["name"],
            "type": obj_type,
            "tactics": "|||".join(data["tactics"]),
            "linked_mitigations": "|||".join(tech_mitigated_by.get(stix_id, [])), 
            "linked_techniques": "|||".join(mitigation_targets.get(stix_id, [])),
            # Store full detection text in metadata for easy retrieval
            "detection_blob": data["detection"] 
        }

        content = f"""
        ID: {data['mitre_id']}
        Name: {data['name']}
        Type: {obj_type}
        Description: {data['description']}
        Tactics: {meta['tactics']}
        """

        documents.append(Document(page_content=content, metadata=meta))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    # vectorstore.delete_collection() 
    vectorstore.add_documents(documents)
    print(f"🚀 Indexed {len(documents)} objects in {COLLECTION_NAME}")

if __name__ == "__main__":
    main()
