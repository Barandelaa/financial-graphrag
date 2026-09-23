from __future__ import annotations

import logging
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_GROQ_MODEL = "mixtral-8x7b-32768"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


def _ollama_available(base_url: str, model: str) -> bool:
    try:
        import ollama

        client = ollama.Client(host=base_url)
        models = [m.get("name", "") or m.get("model", "") for m in client.list().get("models", [])]
        if not models:
            logger.warning("Ollama server reachable but no models pulled")
            return False
        if model not in models and not any(m.startswith(model) for m in models):
            logger.warning(
                "Ollama model '%s' not pulled. Available: %s. "
                "Run: ollama pull %s",
                model,
                ", ".join(sorted(models)) or "none",
                model,
            )
            return False
        return True
    except Exception as exc:
        logger.warning(
            "Ollama server not available at %s (%s); falling back to Groq",
            base_url,
            exc,
        )
        return False


def create_llm(
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    groq_model: str = DEFAULT_GROQ_MODEL,
    prefer_ollama: bool = True,
    num_predict: int = 1200,
    num_ctx: int = 8192,
    keep_alive: str = "10m",
    num_thread: Optional[int] = None,
    timeout: int = 180,
    json_mode: bool = True,
) -> BaseChatModel:
    if prefer_ollama and _ollama_available(ollama_base_url, ollama_model):
        from langchain_ollama import ChatOllama

        logger.info(
            "Using local Ollama model: %s (num_predict=%d, num_ctx=%d, keep_alive=%s)",
            ollama_model,
            num_predict,
            num_ctx,
            keep_alive,
        )
        kwargs: dict = dict(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.0,
            num_predict=num_predict,
            num_ctx=num_ctx,
            keep_alive=keep_alive,
            timeout=timeout,
            # qwen3: desactiva thinking (<think> vacía JSON y explica tus Invalid json output)
            reasoning=False,
        )
        # ReAct necesita tool_calls nativos: sin format=json. Solo el extractor/grafo determinista lo usa.
        if json_mode:
            kwargs["format"] = "json"
        if num_thread is not None:
            kwargs["num_thread"] = num_thread
        try:
            return ChatOllama(**kwargs)
        except TypeError as exc:
            if "reasoning" in str(exc) or "think" in str(exc):
                kwargs.pop("reasoning", None)
                kwargs.pop("think", None)
                logger.debug("ChatOllama no soporta reasoning param, reintenta sin él: %s", exc)
                # último intento con think como alias
                try:
                    return ChatOllama(**kwargs, think=False)  # type: ignore
                except TypeError:
                    return ChatOllama(**kwargs)
            raise

    from langchain_groq import ChatGroq

    logger.info("Using Groq model: %s", groq_model)
    return ChatGroq(model=groq_model, temperature=0.0)


def create_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Optional[Embeddings]:
    try:
        from src.env import get_hf_token
        from langchain_huggingface import HuggingFaceEmbeddings

        token = get_hf_token()
        kwargs = {}
        if token:
            kwargs["token"] = token
        return HuggingFaceEmbeddings(model_name=model_name, **kwargs)
    except Exception as exc:
        logger.warning("Could not create local embeddings (%s)", exc)
        return None
