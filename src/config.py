import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# Load environment variables from .env file
load_dotenv()

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class Config:

    # Pinecone clients
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = "document-index"
    pinecone_client = None
    pinecone_vector_store_client = None

    # OpenAI Clients
    embedding_client = OpenAIEmbeddings(model='text-embedding-3-large')
    chat_client = ChatOpenAI(model_name="gpt-4o", temperature=0.5)

