import re
from mitre_store import get_vectorstore 

vectorstore = get_vectorstore()

class IntentRouter:
    def __init__(self):
        # Strict Cutoff: 1.2 (Lower is better in Chroma/L2)
        # Any semantic match worse than this is ignored unless explicitly requested.
        self.CONFIDENCE_THRESHOLD = 1.2 

    def search_anchor(self, query, type_filter):
        results = vectorstore.similarity_search_with_score(query, k=1, filter={"type": type_filter})
        if not results: return None
        doc, score = results[0]
        # Only use anchor if it's a "Good Match"
        if score > self.CONFIDENCE_THRESHOLD: return None
        return doc

    def solve(self, query):
        q = query.lower()

        # ---------------------------------------------------------
        # INTENT 1: "Defenses for [Technique]" (Separation Update)
        # ---------------------------------------------------------
        if any(x in q for x in ["defenses for", "mitigation for", "how to stop", "prevent"]):
            target = q.replace("defenses for", "").replace("mitigation for", "").replace("how to stop", "").replace("prevent", "").strip()
            
            doc = self.search_anchor(target, "attack-pattern")
            if not doc: return f"❓ Could not identify a technique for '{target}' (Low Confidence)."
            
            mid = doc.metadata.get("mitre_id")
            name = doc.metadata.get("name")
            
            # Section A: Prevention / Hardening (M-IDs)
            links = doc.metadata.get("linked_mitigations", "").split("|||")
            mitigation_section = []
            if links and links[0]:
                for link in links:
                    # Heuristic: "Audit" usually implies detection-focused mitigation, but broadly M-IDs are prevention/hardening
                    icon = "🔒" if "Audit" not in link else "📜" 
                    mitigation_section.append(f"   {icon} {link}")
            else:
                mitigation_section.append("   ⚠️ No specific M-IDs listed.")

            # Section B: Detection (The Detection Blob)
            detection_text = doc.metadata.get("detection_blob", "No detection logic available.")
            # Truncate if massive
            if len(detection_text) > 300: detection_text = detection_text[:300] + "..."

            return (
                f"🔎 Context: {name} ({mid})\n"
                f"🛡️ PREVENT & HARDEN (Mitigations):\n" + "\n".join(mitigation_section) + "\n\n"
                f"👁️ DETECT (Analytics):\n   {detection_text}"
            )

        # ---------------------------------------------------------
        # INTENT 2: "Techniques mitigated by [Mitigation]"
        # ---------------------------------------------------------
        if "mitigated by" in q or "prevented by" in q:
            target = q.split("by")[-1].strip()
            doc = self.search_anchor(target, "course-of-action")
            if not doc: return f"❓ Could not identify mitigation '{target}'."
            
            links = doc.metadata.get("linked_techniques", "").split("|||")
            if not links or not links[0]: return f"ℹ️ {doc.metadata.get('name')} has no mapped techniques."
            
            return f"⚔️ Techniques mitigated by {doc.metadata.get('name')}:\n" + "\n".join([f"   🔻 {l}" for l in links])

        # ---------------------------------------------------------
        # INTENT 3: "List techniques under [Tactic]" (Dominance Update)
        # ---------------------------------------------------------
        tactic_map = [
            "credential access", "execution", "persistence", "privilege escalation", 
            "defense evasion", "discovery", "lateral movement", "collection", 
            "command and control", "exfiltration", "impact", "initial access"
        ]
        found_tactic = next((t for t in tactic_map if t in q), None)
        
        if found_tactic and "list" in q:
            # We fetch many, but strictly filter
            results = vectorstore.similarity_search(found_tactic, k=100, filter={"type": "attack-pattern"})
            valid_hits = []
            
            for doc in results:
                # STRICT DOMINANCE CHECK: The tactic MUST be in the metadata list
                doc_tactics = doc.metadata.get("tactics", "").split("|||")
                if found_tactic in doc_tactics:
                    valid_hits.append(f"[{doc.metadata.get('mitre_id')}] {doc.metadata.get('name')}")
            
            if not valid_hits: return f"⚠️ No techniques found explicitly tagged with '{found_tactic}'."
            return f"📂 Techniques under '{found_tactic.title()}':\n" + "\n".join([f"   🔸 {h}" for h in sorted(list(set(valid_hits)))[:15]])

        # ---------------------------------------------------------
        # INTENT 4: ID Lookup
        # ---------------------------------------------------------
        id_match = re.search(r"\b([TM]\d{4}(?:\.\d{3})?)\b", query.upper())
        if id_match:
            results = vectorstore.similarity_search(query, k=1, filter={"mitre_id": id_match.group(1)})
            if results:
                return f"📄 {results[0].metadata.get('mitre_id')} - {results[0].metadata.get('name')}\n   {results[0].page_content[:200]}..."

        # ---------------------------------------------------------
        # DEFAULT: Semantic Search (Capping Update)
        # ---------------------------------------------------------
        results = vectorstore.similarity_search_with_score(query, k=3, filter={"type": "attack-pattern"})
        
        # Filter out bad scores
        good_matches = [
            f"   💀 [{doc.metadata.get('mitre_id')}] {doc.metadata.get('name')}" 
            for doc, score in results if score < self.CONFIDENCE_THRESHOLD
        ]
        
        if not good_matches:
            # Explicit Rejection
            return "❌ No high-confidence matches found. Please specify a Tactic, ID, or valid Technique name."
            
        return "🔎 Semantic Matches:\n" + "\n".join(good_matches)

router = IntentRouter()

if __name__ == "__main__":
    queries = [
    "Defenses for Keylogging",
    "What mitigations apply to T1056.001 (Keylogging)?",
    "Defenses for GUI Input Capture (T1056.002)",
    "Mitigations for Web Portal Capture (T1056.003)",
    "Does Credential Access Protection (M1043) mitigate Keylogging?",
    "Is Keylogging (T1056.001) primarily Credential Access or Collection?",

    "Techniques mitigated by Privileged Process Integrity",
    "Which techniques are mitigated by Credential Access Protection (M1043)?",
    "Which techniques are mitigated by Privileged Account Management (M1026)?",
    "Which techniques are mitigated by Audit (M1047)?",

    "List techniques under Credential Access",
    "List sub-techniques of OS Credential Dumping (T1003)",
    "Is T1003.001 a sub-technique of OS Credential Dumping?",
    "Should parent techniques be listed when sub-techniques are present?",

    "How do adversaries avoid detection?",
    "Is OS Credential Dumping (T1003) ever considered Collection?",
    "Is Modify Authentication Process (T1556) Credential Access or Defense Evasion?",
    "Is Input Capture (T1056) ever considered Collection?",
    "Which techniques belong to both Credential Access and Defense Evasion?",

    "An attacker installs a malicious SSP DLL. Which ATT&CK techniques apply?",
    "An attacker dumps LSASS using a signed driver. Which ATT&CK techniques apply?",
    "An attacker captures credentials via a fake login page. Which ATT&CK techniques apply?",
    "An attacker modifies authentication packages to bypass MFA. Which ATT&CK techniques apply?",
    "An attacker uses stolen credentials without dumping them. Which ATT&CK techniques apply?",

    "Which mitigation directly prevents Credential Stuffing (T1110.004)?",
    "Which mitigations detect Credential Stuffing but do not prevent it?",
    "Is Audit (M1047) a prevention or detection control?",
    "Which mitigations reduce blast radius rather than stop attacks?",

    "Is SSL/TLS Inspection (M1020) a mitigation for Credential Access?",
    "Which Credential Access techniques lack explicit mitigations in ATT&CK?",
    "Does Input Capture always imply Credential Access?",
    "Is Credential Stuffing possible without prior Credential Access?",
    "Can OS Credential Dumping occur without LSASS?"
]

    for q in queries:
        print(f"\n❓ QUERY: {q}")
        print(router.solve(q))


