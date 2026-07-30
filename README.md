# Text RAG System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![React](https://img.shields.io/badge/React-Frontend-61dafb)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

A full-stack Retrieval-Augmented Generation website for uploading ZIP archives of `.txt` documents, building isolated searchable workspaces, and asking questions grounded in the uploaded content.

The repository separates the user interface from the original RAG implementation:

```text
📦 RAG_practice/
├── 📁 frontend/              # React + Vite web interface
├── 📁 backend/               # FastAPI and RAG engine
├── 📁 storage/               # User workspaces and ChromaDB
├── 📁 docs/                  # Project documentation
├── 🐳 docker-compose.yml     
├── ⚙️ .env        
├── 🚫 .gitignore             
└── 📄 README.md              
```

## 📖 Project Overview

Users can:

1. Upload a ZIP archive containing one or more `.txt` files.
2. Create an isolated RAG workspace for that upload.
3. Select retrieval.
4. Select response generator.
5. Ask questions about the uploaded documents.
6. Inspect the retrieved chunks, source paths, sections, scores, and citations.
7. Keep multiple workspaces or delete them independently.

## 🧰 Tech Stack

**Frontend:** React, Vite, JavaScript, CSS, Node.js  
**Tools:** nvm, npm  
**Backend API:** FastAPI, Pydantic, Uvicorn  
**RAG backend:** Python, PyTorch, Sentence Transformers, BM25  
**Storage:** ChromaDB, JSON manifests, local filesystem  
**LLM server:** Ollama  
**Deployment:** Docker, Docker Compose, Nginx

## 🎥 Demo
- Watch the demo [here](./demo.gif)

<img src="./demo.gif">

## ⚡ Quick Start

```bash
git clone ...
cd project

docker compose up --build
```
Now open `http://localhost:8080`

> [NOTE]
> Docker Compose automatically starts the frontend and backend.

> [TIP]
> Use Ollama for local LLM generation.

> [WARNING]
> Workspace deletion is permanent.
