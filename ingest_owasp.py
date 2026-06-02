"""
Build OWASP vector DB on your PC (needs ~1–2 GB RAM), then deploy to Render.

  pip install -r requirements.txt
  python ingest_owasp.py
  git add chroma_db/owasp
  git commit -m "Add OWASP chroma index"
  git push
"""

import sys

from owasp_ingest import build_owasp_db


def main():
    print("Building OWASP index locally (do not run on Render 512MB)...")
    try:
        n = build_owasp_db(force=True)
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)
    print(f"🎉 {n} chunks saved to ./chroma_db/owasp")
    print("Push chroma_db/owasp to GitHub, then redeploy Render.")


if __name__ == "__main__":
    main()
