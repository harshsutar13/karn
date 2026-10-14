"""One-shot ingestion of P-101 docs into Qdrant."""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.ingestion import ingest_document
from rag.vector_store import get_client, create_collection_if_not_exists
from config import QDRANT_COLLECTION

DOC_DIR = r"D:\SIH26\backend\docs"

def main():
    paths = sorted(glob.glob(os.path.join(DOC_DIR, "*.txt")))
    if not paths:
        print(f"[!] No documents in {DOC_DIR}")
        return

    print(f"Found {len(paths)} documents. Ensuring collection...")
    client = get_client()
    create_collection_if_not_exists(client, QDRANT_COLLECTION)
    print(f"[OK] Collection '{QDRANT_COLLECTION}' ready.\n")

    ok = fail = 0
    for i, p in enumerate(paths, 1):
        name = os.path.basename(p)
        print(f"[{i}/{len(paths)}] {name}")
        try:
            r = ingest_document(p)
            print(f"    -> {r.chunks_upserted} chunks in {r.elapsed_total_s:.1f}s")
            ok += 1
        except Exception as e:
            print(f"    XX {type(e).__name__}: {e}")
            fail += 1

    print(f"\nDone. {ok} ok, {fail} failed.")

if __name__ == "__main__":
    main()