"""CLI: python ingest_owasp.py"""

import sys

from owasp_ingest import build_owasp_db


def main():
    print("Building OWASP Chroma DB from GitHub (2021/docs/en)...")
    try:
        n = build_owasp_db(force=True)
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)
    print(f"🎉 SUCCESS: {n} chunks in ./chroma_db/owasp")
    print("Commit chroma_db/owasp to git OR redeploy Render (build runs ingest).")


if __name__ == "__main__":
    main()
