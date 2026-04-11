"""FastAPI routes wrapping Colophon's CLI functions."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel


# Request models
class AnalyzeRequest(BaseModel):
    document_json: dict[str, Any]
    lang: str = "en"
    analyzers: list[str] | None = None


class StylizeRequest(BaseModel):
    article_text: str
    style_profile_json: dict[str, Any]
    source_text: str | None = None
    provider: str | None = None
    model: str | None = None
    no_diff: bool = False


class IndexRequest(BaseModel):
    profile_json: dict[str, Any]
    author: str
    lang: str = "en"


class SearchRequest(BaseModel):
    text: str
    lang: str = "en"
    top: int = 10


def create_router(*, db_url: str | None = None) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.post("/ingest")
    async def ingest_file(file: UploadFile = File(...), lang: str = "en"):
        """Upload and ingest a document."""
        from colophon.ingestion import ingest

        suffix = Path(file.filename or "upload.txt").suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        doc = ingest(tmp_path)
        tmp_path.unlink(missing_ok=True)
        return doc.model_dump(mode="json")

    @router.post("/analyze")
    async def analyze_document(body: AnalyzeRequest):
        """Run stylometric analysis on an ingested document."""
        from colophon.analysis.pipeline import analyze
        from colophon.models.document import Document

        doc = Document.model_validate(body.document_json)
        profile = analyze(doc, analyzers=body.analyzers, lang=body.lang)
        return profile.model_dump(mode="json")

    @router.post("/stylize")
    async def stylize_article(body: StylizeRequest):
        """Restyle an article in a target author's voice."""
        from colophon.models.features import StyleProfile
        from colophon.stylize.llm import stylize_text
        from colophon.stylize.prompt import build_style_prompt

        profile = StyleProfile.model_validate(body.style_profile_json)

        # Build few-shot examples if source text provided
        examples = None
        if body.source_text:
            from colophon.models.document import Document
            from colophon.stylize.examples import select_examples
            source_doc = Document.from_text(text=body.source_text, source_path="source")
            examples = select_examples(source_doc, profile)

        # Build fingerprint diff
        fingerprint_diff = None
        if not body.no_diff:
            try:
                from colophon.analysis.pipeline import analyze
                from colophon.models.document import Document
                from colophon.stylize.diff import compute_fingerprint_diff
                input_doc = Document.from_text(text=body.article_text, source_path="input")
                input_profile = analyze(
                    input_doc,
                    analyzers=["sentences", "contractions", "paragraphs", "punctuation",
                               "sentence_openers", "dialogue"],
                )
                fingerprint_diff = compute_fingerprint_diff(input_profile, profile)
            except Exception:
                pass

        system_prompt = build_style_prompt(profile, examples=examples, fingerprint_diff=fingerprint_diff)
        result = stylize_text(system_prompt, body.article_text, provider=body.provider, model=body.model)
        return {
            "restyled": result,
            "original_words": len(body.article_text.split()),
            "restyled_words": len(result.split()),
        }

    @router.post("/stylize/prompt")
    async def preview_prompt(body: StylizeRequest):
        """Preview the system prompt without calling the LLM."""
        from colophon.models.features import StyleProfile
        from colophon.stylize.prompt import build_style_prompt

        profile = StyleProfile.model_validate(body.style_profile_json)
        prompt = build_style_prompt(profile)
        return {"prompt": prompt}

    @router.post("/index")
    async def index_document(body: IndexRequest):
        """Index a style profile into pgvector."""
        if not db_url:
            return {"error": "No database configured. Start with --db flag."}

        from colophon.db import get_session
        from colophon.db.operations import index_profile
        from colophon.embeddings.vectorize import style_profile_to_vector
        from colophon.models.features import StyleProfile

        profile = StyleProfile.model_validate(body.profile_json)
        vector = style_profile_to_vector(profile)

        session = get_session(db_url)
        try:
            row = index_profile(session, profile, vector, author=body.author, lang=body.lang)
            return {"id": row.id, "author": body.author, "title": profile.document_title}
        finally:
            session.close()

    @router.post("/search")
    async def search_style(body: SearchRequest):
        """Search for similar authors and documents."""
        if not db_url:
            return {"error": "No database configured. Start with --db flag."}

        from colophon.analysis.pipeline import analyze
        from colophon.db import get_session
        from colophon.db.operations import search_authors, search_similar
        from colophon.embeddings.vectorize import style_profile_to_vector
        from colophon.models.document import Document

        doc = Document.from_text(text=body.text, source_path="search")
        profile = analyze(doc, lang=body.lang)
        vector = style_profile_to_vector(profile)

        session = get_session(db_url)
        try:
            authors = search_authors(session, vector, limit=body.top)
            documents = search_similar(session, vector, limit=body.top)
            return {"authors": authors, "documents": documents}
        finally:
            session.close()

    @router.get("/corpus")
    async def list_corpus():
        """List all indexed authors and documents."""
        if not db_url:
            return {"authors": [], "documents": []}

        from colophon.db import get_session
        from colophon.db.schema import AuthorProfileRow, StyleProfileRow

        session = get_session(db_url)
        try:
            authors = session.query(AuthorProfileRow).order_by(AuthorProfileRow.author).all()
            docs = session.query(StyleProfileRow).order_by(StyleProfileRow.document_title).all()
            return {
                "authors": [{"author": a.author, "doc_count": a.doc_count} for a in authors],
                "documents": [
                    {
                        "id": d.id,
                        "title": d.document_title,
                        "author": d.author,
                        "lang": d.lang,
                        "word_count": d.word_count,
                    }
                    for d in docs
                ],
            }
        finally:
            session.close()

    @router.get("/languages")
    async def list_languages():
        """Return supported language profiles."""
        from colophon.lang import PROFILES
        return [{"code": code, "name": p.name} for code, p in PROFILES.items()]

    @router.get("/providers")
    async def list_providers():
        """Return available LLM providers (based on .env keys)."""
        import os

        from dotenv import load_dotenv
        load_dotenv()

        providers = []
        checks = [
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("openai", "OPENAI_API_KEY"),
            ("gemini", "GEMINI_API_KEY"),
            ("openrouter", "OPENROUTER_API_KEY"),
        ]
        for name, env_var in checks:
            if os.environ.get(env_var):
                providers.append(name)
        return providers

    return router
