import logging
import os

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

import app.config  # noqa: F401 — carga variables desde .env

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")


def get_qdrant_client():
    try:
        client = QdrantClient(url=QDRANT_URL)
        return client
    except Exception as e:
        logging.error(f"Could not connect to Qdrant: {e}")
        return None


def init_qdrant_collection(collection_name="initiatives"):
    client = get_qdrant_client()
    if not client:
        return

    try:
        collections_to_init = ["initiatives", "strategic_docs"]

        existing_collections = [c.name for c in client.get_collections().collections]

        for coll in collections_to_init:
            if coll not in existing_collections:
                client.create_collection(
                    collection_name=coll,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                logging.info(f"Created Qdrant collection: {coll}")

    except Exception as e:
        logging.error(f"Failed to initialize Qdrant collections: {e}")
