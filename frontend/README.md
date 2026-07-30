# Text RAG Frontend
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)
![Vite](https://img.shields.io/badge/Vite-7-646CFF?logo=vite)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6-F7DF1E?logo=javascript)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)

React and Vite frontend interacting with FastAPI backend

## 🧰 Tech Stack

**Framework:** React  
**Build tool:** Vite  
**Language:** JavaScript and JSX  
**Styling:** Plain CSS  
**Production server:** Nginx  
**Container:** Docker  

## 📁 Frontend Structure

```text
frontend/
├── src/
│   ├── components/
│   │   ├── ChatPanel.jsx
│   │   ├── SourceCard.jsx
│   │   ├── UploadPanel.jsx
│   │   └── WorkspacePanel.jsx
│   ├── api.js
│   ├── App.jsx
│   ├── main.jsx
│   └── styles.css
├── .env
├── Dockerfile
├── index.html
├── nginx.conf
├── package.json
└── vite.config.js
```

## ✨ Features

The frontend handles:

- API health checking.
- ZIP selection and drag-and-drop.
- Workspace creation.
- Workspace listing and selection.
- Workspace deletion confirmation.
- Retriever, generator, and `top_k` controls.
- Question submission.
- Chat-style response rendering.
- Answerability status.
- Citations and retrieved-chunk inspection.
- Loading, success, and error states.
- Responsive layout.
- Remembering the selected workspace ID

## 📦 Install Dependencies

Install dependencies from `package.json`

```bash
npm install
```

## 💻 Run in Development

Ensure the backend is running at `http://localhost:8000`, then run:

```bash
npm run dev
```

Open:

```text
http://localhost:5173
```

The Vite configuration proxies:

```text
/api/* → http://localhost:8000/api/*
```

This avoids cross-origin configuration during normal local development.

## 🏗️ Build for Production

```bash
npm run build
```

The generated static files are written to:

```text
dist/
```

Preview the production build locally:

```bash
npm run preview
```

## 🐳 Docker Build

The frontend Dockerfile uses two stages:

1. Node.js installs dependencies and runs `vite build`.
2. Nginx serves the generated static files.

From the repository root:

```bash
docker compose up --build
```

The frontend is then available at:

```text
http://localhost:8080
```

## 🎨 Styling

All interface styling is in `src/styles.css`.

