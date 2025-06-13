from opensearchpy import OpenSearch, RequestsHttpConnection
from dotenv import load_dotenv
from tools.contants import INDEX_NAME, OPENSEARCH_HOST, OPENSEARCH_PORT
import os

load_dotenv()


def main(index_name: str):
    """
    Create index in OpenSearch (without embeddings)

    Args:
        index_name (str): Index name

    Returns:
        None
    """

    host = OPENSEARCH_HOST
    port = OPENSEARCH_PORT

    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        # http_auth=("admin", "admin"),
        use_ssl=False,
        verify_certs=False,
        connection_class=RequestsHttpConnection,
    )

    # Mapping index
    if not client.indices.exists(index=index_name):
        MAPPING = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "text": {"type": "text"},
                    "vector_field": {
                        "type": "knn_vector",
                        "dimension": 1536,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                        },
                    },
                }
            },
        }

        # Index creation
        if not client.indices.exists(index=index_name):
            response = client.indices.create(index=index_name, body=MAPPING)
            print(f"Index '{index_name}' created successfully:\n{response}")
        else:
            print(f"Index '{index_name}' already exists.")


if __name__ == "__main__":
    main(index_name=INDEX_NAME)
