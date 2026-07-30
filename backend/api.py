from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import sys
import threading
import uuid
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_project.config import load_config
from rag_project.embeddings.sentence_transformer import SentenceTransformerEmbeddingModel
from rag_project.factory import create_generator, create_retriever
from rag_project.loaders import TextDirectoryLoader
from rag_project.pipeline import RAGPipeline
from rag_project.preprocessing import SectionChunker, SphinxTextPreprocessor
from rag_project.storage.chroma_store import ChromaVectorStore

load_dotenv(PROJECT_ROOT / ".env")

WEB_DATA_ROOT = Path(os.getenv("WEB_DATA_ROOT", PROJECT_ROOT / "web_data")).resolve()
BASE_CONFIG_PATH = Path(os.getenv("RAG_BASE_CONFIG", PROJECT_ROOT / "configs" / "sources.json")).resolve()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_UNCOMPRESSED_MB = int(os.getenv("MAX_UNCOMPRESSED_MB", "200"))
MAX_UNCOMPRESSED_BYTES = MAX_UNCOMPRESSED_MB * 1024 * 1024
MAX_TEXT_FILES = int(os.getenv("MAX_TEXT_FILES", "2000"))
ALLOWED_ORIGINS = [value.strip() for value in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:8080").split(",") if value.strip()]
WEB_DATA_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Uploaded Text RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_PIPELINES: dict[tuple[str, str, str, int], RAGPipeline] = {}
_PIPELINE_LOCK = threading.Lock()
_WORKSPACE_LOCKS: dict[str, threading.Lock] = {}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    retriever: Literal["bm25", "dense"] = "bm25"
    generator: Literal["extractive", "openai-compatible"] = "extractive"
    top_k: int = Field(default=5, ge=1, le=20)


class CitationResponse(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    relative_path: str
    section_path: list[str]
    excerpt: str


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    rank: int
    score: float
    retriever: str
    relative_path: str
    section_path: list[str]
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    answerable: bool
    citations: list[CitationResponse]
    retrieved: list[RetrievedChunkResponse]
    metadata: dict


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _workspace_dir(workspace_id: str) -> Path:
    if not workspace_id or any(character not in "0123456789abcdef" for character in workspace_id):
        raise HTTPException(status_code=404, detail="Workspace not found")
    path = (WEB_DATA_ROOT / workspace_id).resolve()
    if path.parent != WEB_DATA_ROOT:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return path


def _manifest_path(workspace_id: str) -> Path:
    return _workspace_dir(workspace_id) / "manifest.json"


def _read_manifest(workspace_id: str) -> dict:
    path = _manifest_path(workspace_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_base_config() -> dict:
    if not BASE_CONFIG_PATH.exists():
        raise RuntimeError(f"Base RAG config does not exist: {BASE_CONFIG_PATH}")
    return load_config(BASE_CONFIG_PATH)


def _workspace_config(workspace_id: str, title: str, raw_dir: Path, vector_dir: Path) -> dict:
    base = _load_base_config()
    return {
        "docs_root": str(raw_dir),
        "documents": [
            {
                "document_id": f"upload_{workspace_id[:12]}",
                "title": title,
                "include": ["**/*.txt"],
                "exclude": [],
            }
        ],
        "preprocessing": dict(base.get("preprocessing", {"remove_navigation_lines": True})),
        "chunking": dict(base["chunking"]),
        "embedding": dict(base["embedding"]),
        "vector_store": {
            "path": str(vector_dir),
            "collection": f"uploaded_text_{workspace_id[:16]}",
        },
    }


async def _save_upload(upload: UploadFile, destination: Path) -> int:
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"ZIP file exceeds the {MAX_UPLOAD_MB} MB upload limit")
            output.write(chunk)
    return size


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o170000
    return mode == stat.S_IFLNK


def _safe_extract_text_files(zip_path: Path, raw_dir: Path) -> list[str]:
    extracted: list[str] = []
    total_uncompressed = 0
    raw_root = raw_dir.resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as error:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid ZIP archive") from error
    with archive:
        candidates = [info for info in archive.infolist() if not info.is_dir() and not info.filename.startswith("__MACOSX/")]
        text_candidates = [info for info in candidates if PurePosixPath(info.filename).suffix.lower() == ".txt"]
        if not text_candidates:
            raise HTTPException(status_code=400, detail="The ZIP archive does not contain any .txt files")
        if len(text_candidates) > MAX_TEXT_FILES:
            raise HTTPException(status_code=400, detail=f"The ZIP contains more than {MAX_TEXT_FILES} text files")
        for info in text_candidates:
            if _is_zip_symlink(info):
                raise HTTPException(status_code=400, detail=f"Symbolic links are not allowed: {info.filename}")
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise HTTPException(status_code=400, detail=f"Unsafe ZIP path: {info.filename}")
            total_uncompressed += info.file_size
            if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
                raise HTTPException(status_code=413, detail=f"Extracted text exceeds the {MAX_UNCOMPRESSED_MB} MB limit")
            destination = (raw_root / Path(*member.parts)).resolve()
            if raw_root not in destination.parents:
                raise HTTPException(status_code=400, detail=f"Unsafe ZIP path: {info.filename}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(destination.relative_to(raw_root).as_posix())
    return sorted(extracted)


@lru_cache(maxsize=4)
def _embedding_model(model_name: str, batch_size: int) -> SentenceTransformerEmbeddingModel:
    return SentenceTransformerEmbeddingModel(model_name=model_name, batch_size=batch_size)


def _build_workspace_index(config: dict) -> dict:
    sources = TextDirectoryLoader().load(config)
    preprocessing = config.get("preprocessing", {})
    preprocessor = SphinxTextPreprocessor(remove_navigation_lines=bool(preprocessing.get("remove_navigation_lines", True)))
    chunking = config["chunking"]
    chunker = SectionChunker(
        max_chars=int(chunking.get("max_chars", 2400)),
        overlap_paragraphs=int(chunking.get("overlap_paragraphs", 1)),
        minimum_chars=int(chunking.get("minimum_chars", 120)),
    )
    sections = []
    for source in sources:
        sections.extend(preprocessor.process(source))
    chunks = chunker.chunk(sections)
    if not chunks:
        raise ValueError("No chunks were created. Check that the text files contain enough readable text.")
    embedding = config["embedding"]
    model = _embedding_model(embedding["model_name"], int(embedding.get("batch_size", 32)))
    vector = config["vector_store"]
    store = ChromaVectorStore(path=vector["path"], collection_name=vector["collection"])
    store.rebuild(chunks, model)
    return {
        "source_file_count": len(sources),
        "section_count": len(sections),
        "chunk_count": len(chunks),
        "embedding_model": embedding["model_name"],
    }


def _get_pipeline(workspace_id: str, retriever_name: str, generator_name: str, top_k: int) -> RAGPipeline:
    key = (workspace_id, retriever_name, generator_name, top_k)
    with _PIPELINE_LOCK:
        cached = _PIPELINES.get(key)
        if cached is not None:
            return cached
    config_path = _workspace_dir(workspace_id) / "config.json"
    if not config_path.exists():
        raise HTTPException(status_code=409, detail="Workspace index configuration is missing")
    config = load_config(config_path)
    retriever = create_retriever(retriever_name, config)
    generator = create_generator(generator_name)
    pipeline = RAGPipeline(retriever=retriever, generator=generator, top_k=top_k)
    with _PIPELINE_LOCK:
        _PIPELINES[key] = pipeline
    return pipeline


def _clear_workspace_cache(workspace_id: str) -> None:
    with _PIPELINE_LOCK:
        for key in [key for key in _PIPELINES if key[0] == workspace_id]:
            _PIPELINES.pop(key, None)


def _serialise_response(response) -> AskResponse:
    return AskResponse(
        question=response.question,
        answer=response.answer,
        answerable=response.answerable,
        citations=[CitationResponse(**asdict(citation)) for citation in response.citations],
        retrieved=[
            RetrievedChunkResponse(
                chunk_id=result.chunk.chunk_id,
                rank=result.rank,
                score=result.score,
                retriever=result.retriever,
                relative_path=result.chunk.relative_path,
                section_path=result.chunk.section_path,
                text=result.chunk.text,
            )
            for result in response.retrieved
        ],
        metadata=response.metadata,
    )


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "base_config": str(BASE_CONFIG_PATH),
        "web_data_root": str(WEB_DATA_ROOT),
        "max_upload_mb": MAX_UPLOAD_MB,
    }


@app.get("/api/workspaces")
def list_workspaces() -> list[dict]:
    manifests: list[dict] = []
    for path in WEB_DATA_ROOT.glob("*/manifest.json"):
        try:
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(manifests, key=lambda item: item.get("created_at", ""), reverse=True)


@app.get("/api/workspaces/{workspace_id}")
def get_workspace(workspace_id: str) -> dict:
    return _read_manifest(workspace_id)


@app.post("/api/workspaces", status_code=201)
async def create_workspace(file: UploadFile = File(...), title: str | None = Form(default=None)) -> dict:
    filename = file.filename or "documents.zip"
    if Path(filename).suffix.lower() != ".zip":
        raise HTTPException(status_code=400, detail="Upload a .zip file containing .txt files")
    workspace_id = uuid.uuid4().hex
    workspace_dir = _workspace_dir(workspace_id)
    raw_dir = workspace_dir / "raw"
    vector_dir = workspace_dir / "vector_db"
    upload_path = workspace_dir / "upload.zip"
    display_title = (title or Path(filename).stem or "Uploaded documents").strip()[:120]
    workspace_dir.mkdir(parents=True, exist_ok=False)
    lock = _WORKSPACE_LOCKS.setdefault(workspace_id, threading.Lock())
    try:
        upload_size = await _save_upload(file, upload_path)
        text_files = await asyncio.to_thread(_safe_extract_text_files, upload_path, raw_dir)
        config = _workspace_config(workspace_id, display_title, raw_dir, vector_dir)
        config_path = workspace_dir / "config.json"
        _write_json(config_path, config)
        manifest = {
            "workspace_id": workspace_id,
            "title": display_title,
            "original_filename": filename,
            "status": "indexing",
            "created_at": _utc_now(),
            "upload_size_bytes": upload_size,
            "text_files": text_files,
            "text_file_count": len(text_files),
        }
        _write_json(workspace_dir / "manifest.json", manifest)
        with lock:
            statistics = await asyncio.to_thread(_build_workspace_index, config)
        manifest.update(statistics)
        manifest["status"] = "ready"
        manifest["indexed_at"] = _utc_now()
        _write_json(workspace_dir / "manifest.json", manifest)
        upload_path.unlink(missing_ok=True)
        return manifest
    except HTTPException:
        shutil.rmtree(workspace_dir, ignore_errors=True)
        raise
    except Exception as error:
        shutil.rmtree(workspace_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to build the RAG index: {error}") from error
    finally:
        await file.close()


@app.post("/api/workspaces/{workspace_id}/ask", response_model=AskResponse)
async def ask_workspace(workspace_id: str, request: AskRequest) -> AskResponse:
    manifest = _read_manifest(workspace_id)
    if manifest.get("status") != "ready":
        raise HTTPException(status_code=409, detail="Workspace is not ready")
    try:
        pipeline = await asyncio.to_thread(
            _get_pipeline,
            workspace_id,
            request.retriever,
            request.generator,
            request.top_k,
        )
        response = await asyncio.to_thread(pipeline.ask, request.question)
        return _serialise_response(response)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"RAG request failed: {error}") from error


@app.delete("/api/workspaces/{workspace_id}")
def delete_workspace(workspace_id: str) -> dict:
    workspace_dir = _workspace_dir(workspace_id)
    if not workspace_dir.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")
    _clear_workspace_cache(workspace_id)
    shutil.rmtree(workspace_dir)
    _WORKSPACE_LOCKS.pop(workspace_id, None)
    return {"deleted": True, "workspace_id": workspace_id}
