import re
from mitre_store import get_vectorstore

DESCRIPTION_MAX_LEN = 500


def _extract_description(doc) -> str:
    """Description from metadata or page_content."""
    if doc.metadata.get("description"):
        text = doc.metadata["description"].strip()
    else:
        match = re.search(
            r"Description:\s*(.+?)(?:\n\s*Tactics:|\Z)",
            doc.page_content,
            re.DOTALL | re.IGNORECASE,
        )
        text = match.group(1).strip() if match else doc.page_content.strip()
    if len(text) > DESCRIPTION_MAX_LEN:
        text = text[:DESCRIPTION_MAX_LEN].rsplit(" ", 1)[0] + "..."
    return text


def _format_technique(doc, score=None) -> str:
    mid = doc.metadata.get("mitre_id", "?")
    name = doc.metadata.get("name", "Unknown")
    desc = _extract_description(doc)
    header = f"💀 [{mid}] {name}"
    if score is not None:
        header += f" (relevance: {score:.2f})"
    return f"{header}\n   📖 {desc}"


class IntentRouter:
    def __init__(self):
        # Strict Cutoff: 1.2 (Lower is better in Chroma/L2)
        # Any semantic match worse than this is ignored unless explicitly requested.
        self.CONFIDENCE_THRESHOLD = 1.2

    @property
    def vectorstore(self):
        return get_vectorstore()

    def search_anchor(self, query, type_filter):
        results = self.vectorstore.similarity_search_with_score(query, k=1, filter={"type": type_filter})
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
            results = self.vectorstore.similarity_search(found_tactic, k=100, filter={"type": "attack-pattern"})
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
            results = self.vectorstore.similarity_search(
                query, k=1, filter={"mitre_id": id_match.group(1)}
            )
            if results:
                return "📄 Technique details:\n\n" + _format_technique(results[0])

        # ---------------------------------------------------------
        # Phishing / named-technique lookup (prefer name match + descriptions)
        # ---------------------------------------------------------
        if "phishing" in q:
            return self._semantic_with_descriptions(
                query, extra_terms=["phishing"], k=5
            )

        # ---------------------------------------------------------
        # DEFAULT: Semantic search with descriptions
        # ---------------------------------------------------------
        return self._semantic_with_descriptions(query, k=4)

    def _semantic_with_descriptions(
        self, query: str, k: int = 4, extra_terms: list | None = None
    ) -> str:
        results = self.vectorstore.similarity_search_with_score(
            query, k=k * 3, filter={"type": "attack-pattern"}
        )
        q_lower = query.lower()
        terms = set(re.findall(r"[a-z]{4,}", q_lower))
        if extra_terms:
            terms.update(extra_terms)

        ranked: list[tuple] = []
        for doc, score in results:
            if score >= self.CONFIDENCE_THRESHOLD:
                continue
            name = (doc.metadata.get("name") or "").lower()
            # Prefer techniques whose name contains query terms (e.g. "phishing")
            name_boost = -0.15 if any(t in name for t in terms) else 0.0
            ranked.append((score + name_boost, doc, score))

        ranked.sort(key=lambda x: x[0])
        ranked = ranked[:k]

        if not ranked:
            return (
                "❌ No high-confidence matches found. "
                "Try a MITRE ID (e.g. T1566) or a more specific technique name."
            )

        blocks = [_format_technique(doc, score=raw) for _, doc, raw in ranked]
        return "🔎 MITRE ATT&CK matches:\n\n" + "\n\n".join(blocks)

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


