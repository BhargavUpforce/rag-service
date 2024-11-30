# RAG-Service

A **FastAPI-based backend** designed for **file uploads** and **real-time chat** using **RAG (Retrieval-Augmented Generation)** architecture with **Pinecone** and **OpenAI** for knowledge-driven responses.

---

## 📋 Features

- **File Uploads**:
  - Supports `.pdf`, `.docx`, and `.txt` file uploads.
  - Extracts, cleans unwanted text, and stores chunks of uploaded files in Pinecone for efficient retrieval.
  - Automatically sets up the Pinecone environment when the server starts, creating necessary indexes and configurations for seamless operation.


- **Real-time Chat**:
  - WebSocket-based chat for live, interactive conversations.
  - Performs similarity searches using Pinecone to retrieve contextual information.
  - Leverages OpenAI for generating contextually relevant responses.
  - Maintains chat history for the **current session only** to ensure focused interactions.


## 🛠️ Requirements

- **Python**: Version 3.10
- **Environment Variables**: Create a `.env` file in the root directory with the following keys:
  ```env
  PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
  OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
  ```



## 🚀 Installation and Setup

Follow these steps to set up the project locally:

### 1. Clone the Repository:
```bash
git clone https://github.com/BhargavUpforce/rag-service.git
cd rag-service
```

### 2. Install Dependencies:
Create and activate a virtual environment, then install the required dependencies using:
```bash
pip install -r requirements.txt
```


### 3. Start the Server:
```bash
uvicorn src.app:app --reload
```

### 4. Access API Documentation:
Open your browser and navigate to fastapi docs to explore the available endpoints.
```bash
http://127.0.0.1:8000/docs
```

### 5. WebSocket Testing:
Use Postman WebSocket Collection or similar tools to interact with the WebSocket endpoint:
```bash
ws://127.0.0.1:8000/chat
```


## 📂 Project Structure

```bash
rag-service/
├── src/
│   ├── app.py          # Starting point of the application
│   ├── service.py      # Main service logic
│   ├── utils.py        # Utility functions
│   ├── schema.py       # Input/response schemas
│   └── config.py       # Configuration and client setup
├── .env                # Environment variables
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```


## 🏗️ Potential Improvements

- Refactor the project into smaller, dedicated modules within the `src/` directory to improve maintainability and scalability.
- Organize the code into separate routes for each module to enhance code separation and ease of management.
- **Note**: As this is a small project, these improvements are optional and may be implemented in the future as the project grows.


## 🌟 License

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for more details.


## 📬 Contact

For any questions, collaboration, or feedback, feel free to reach out to the project author:

- **Author**: Bhargav Patel
- **GitHub**: [BhargavUpforce](https://github.com/BhargavUpforce)
