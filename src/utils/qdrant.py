from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from qdrant_client import QdrantClient, models
from decouple import config

vector_store = None
class VectorStore:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.QDRANT_URL = config("QDRANT_URL")
        self.QDRANT_API_KEY = config("QDRANT_API_KEY")
        self.fastembed = FastEmbedEmbeddings()
        self.sparse_embed = FastEmbedSparse()
        self.client = QdrantClient(url=self.QDRANT_URL, api_key=self.QDRANT_API_KEY)
        self.initialize_vector_store()

    def initialize_vector_store(self):
        if not self.client.collection_exists(self.collection_name):
            print(f"Collection {self.collection_name} does not exist. Please create the collection first.")
        else:
            global vector_store
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.fastembed,
                sparse_embedding=self.sparse_embed,
                retrieval_mode=RetrievalMode.HYBRID,
            )
            print(f"Collection {self.collection_name} initialized")

    def create_collection(self, documents):
        """
        Create a collection if it doesn't exist and embed/upload documents.

        Args:
        documents (list): The list of documents to embed and upload

        Returns:
        None
        """
        if not self.client.collection_exists(self.collection_name):
            self.embed_and_upload_documents(documents)
            print(f"Collection {self.collection_name} created")
        else:
            print(f"Collection {self.collection_name} already exists")

    def embed_and_upload_documents(self, documents):
        """
        Embed and upload documents to the collection.

        Args:
        documents (list): The list of documents to embed and upload

        Returns:
        None
        """
        global vector_store
        vector_store = QdrantVectorStore.from_documents(
            documents=documents,
            collection_name=self.collection_name,
            url=self.QDRANT_URL,
            api_key=self.QDRANT_API_KEY,
            embedding=self.fastembed,
            sparse_embedding=self.sparse_embed,
            retrieval_mode=RetrievalMode.HYBRID,
        )

    def test_query_vector_store(self, query):
        """
        Test the query vector store.

        Args:
        query (str): The query to test

        Returns:
        None
        """
        global vector_store
        results = vector_store.similarity_search(query)
        for res in results:
            md = res.metadata
            print(f'SOURCE: {md["source"]}')
            print(f'TITLE: {md["document_name"]}')
            print(f'PAGE: {md["page"]}')
            print(res.page_content)
            print("-------------\n")

    def search(self, text: str):
        """
        Search the vector store.

        Args:
        text (str): The search query

        Returns:
        list: The search results
        """
        search_result = self.vector_store.similarity_search_with_score(
            text, 
            k=10, 
            hybrid_fusion=models.FusionQuery(fusion=models.Fusion.RRF), 
            score_threshold=0.4
        )
        print(search_result)
        return search_result
