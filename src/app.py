from typing import List, Dict
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from starlette.websockets import WebSocket, WebSocketDisconnect

from src.config import Config
from fastapi.responses import JSONResponse
from langchain_core.documents import Document

from src.schema import UploadResponse
from src.services import WebsocketManager, PineconeService
from src.utils import llm_chat_chain, extract_pdf_text, extract_txt_text, extract_docx_text, preprocess_text, chunk_text

# Initialize FastAPI app
app = FastAPI(
    title="RAG Service",
    description="File upload and live chat service using Retrieval-Augmented Generation.",
    version="1.0.0"
)


# Initialize Pinecone during startup
@app.on_event("startup")
async def startup_event():
    """
    Initialize Pinecone index on FastAPI server startup.
    """
    pinecone_service = PineconeService(pinecone_api_key=Config.PINECONE_API_KEY, index_name=Config.INDEX_NAME,
                                       embedding_client=Config.embedding_client)
    # Initialize Pinecone index and store it in Config
    if Config.pinecone_client is None:
        Config.pinecone_client = pinecone_service.initialize_pinecone()

    # Initialize Pinecone vector client
    if Config.pinecone_vector_store_client is None:
        Config.pinecone_vector_store_client = pinecone_service.pinecone_vector_store_client(Config.pinecone_client)


# Root endpoint
@app.get("/")
async def root():
    return JSONResponse(content={"message": "Welcome to the RAG Service API!"})


@app.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload and process a file",
    description="This endpoint allows users to upload a file. The content is extracted, "
                "preprocessed, split into chunks, and stored in Pinecone for vector search.",
    tags=["File Processing"],
)
async def upload_file(file: UploadFile = File(...)):
    """
    Handles file uploads, processes the content, and stores it in Pinecone.

    Args:
    - file: The file to be uploaded. Supported formats are `.pdf`, `.docx`, and `.txt`.

    Returns:
    - A response containing a success message and the IDs of the stored chunks in Pinecone.

    Raises:
    - 400: If the uploaded file type is unsupported.
    """
    # Read the uploaded file content
    file_content = await file.read()

    # Extract text from the file
    extracted_text = ""
    if file.filename.endswith(".pdf"):
        extracted_text = extract_pdf_text(file_content)
    elif file.filename.endswith(".docx"):
        extracted_text = extract_docx_text(file_content)
    elif file.filename.endswith(".txt"):
        extracted_text = extract_txt_text(file_content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Preprocess the extracted text
    preprocessed_text = preprocess_text(extracted_text)

    # Split the preprocessed text into chunks
    text_chunks = chunk_text(preprocessed_text)

    # Store each chunk as a vector in Pinecone
    document_ids = []
    documents = []
    for chunk in text_chunks:
        documents.append(
            Document(
                page_content=chunk,
                metadata={"source": file.filename},
            )
        )
        document_ids.append(str(uuid4()))

    response = Config.pinecone_vector_store_client.add_documents(documents=documents, ids=document_ids)

    return {
        "message": "File uploaded, content extracted, and stored in Pinecone successfully!",
        "stored_chunks_ids": response,
    }


# Websocket Endpoint for chat
websocket_manager = WebsocketManager()


@app.websocket("/chat")
async def chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat.

    This endpoint allows real-time communication between the client and the server using WebSocket.
    Clients can send a message, and the server processes it to:
    - Perform a similarity search using Pinecone to retrieve context from a vector store.
    - Include historical chat context for improved responses.
    - Generate a response using the `llm_chat_chain` function.

    Args:
    - websocket (WebSocket): The WebSocket connection object.

    Workflow:
    1. The client sends a message to the server.
    2. The server retrieves relevant context from Pinecone.
    3. The server processes the message along with the retrieved context and chat history.
    4. The server sends the generated response back to the client.
    5. The conversation history is updated.

    Raises:
    - WebSocketDisconnect: When the WebSocket connection is closed by the client.

    Note:
    - This endpoint is designed for real-time interactions and will not appear in FastAPI's OpenAPI documentation.
    """
    await websocket_manager.connect(websocket)
    history_data: Dict[str, str] = {}  # Stores conversation history

    try:
        while True:
            # Receive the user's message
            message = await websocket.receive_text()

            # Perform a similarity search to fetch relevant context
            results = Config.pinecone_vector_store_client.similarity_search(
                message,
                k=2  # Retrieve top 2 most similar results
            )

            # Build context from the search results
            context = ""
            for count, res in enumerate(results, 1):
                context += f"Context {count}: {res.page_content}\n\n"

            # Build chat history
            history = ""
            for count, (question, response) in enumerate(history_data.items(), 1):
                history += f"({count})\nQuestion: {question}\nResponse: {response}\n\n"

            # Generate a response using the LLM chat chain
            response = llm_chat_chain(query=message, history=history, context=context)

            # Update conversation history
            history_data[message] = response

            # Send the response back to the client
            await websocket_manager.send_message(response, websocket)
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
