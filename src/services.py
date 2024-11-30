from fastapi import WebSocket
from typing import List
import time

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec, Index


class WebsocketManager:
    """
    Manages WebSocket connections.
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


class PineconeService:
    """
    Manages Pinecone connections and setup.
    """

    def __init__(self, pinecone_api_key: str, index_name: str, embedding_client: OpenAIEmbeddings):
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.embedding_client = embedding_client

    def initialize_pinecone(self):
        """
        Ensures the Pinecone index is created and initialized.
        """
        pc = Pinecone(api_key=self.pinecone_api_key)

        # Check if the index exists
        if self.index_name not in pc.list_indexes().names():
            print(f"Creating Pinecone index: {self.index_name}...")
            pc.create_index(
                self.index_name,
                dimension=3072,  # For text-embedding-3-large model
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait for the index to be ready
            while not pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
            print(f"Index {self.index_name} is ready.")
        else:
            print(f"Index {self.index_name} already exists.")

        # Return the Pinecone index instance
        return pc.Index(self.index_name)

    def pinecone_vector_store_client(self, pinecone_client: Index):
        """
        Creates Pinecone vector client.
        """
        return PineconeVectorStore(index=pinecone_client, embedding=self.embedding_client)


class OpenAIService:
    """
    Manages OpenAI connections and setup.
    """




