from __future__ import annotations

import sys
import types


class ChatVertexAI:
    """Compat stub. langchain-community 0.4.x removed the vertexai
    chat model module, but ragas 0.4.3 imports it at module load time
    and only uses it for isinstance checks."""


class VertexAI:
    """Compat stub for langchain_community.llms.VertexAI."""


def install() -> None:
    import langchain_community

    if hasattr(langchain_community, "chat_models"):
        chat_models = langchain_community.chat_models
    else:
        chat_models = types.ModuleType("langchain_community.chat_models")
        langchain_community.chat_models = chat_models
        sys.modules["langchain_community.chat_models"] = chat_models

    if not hasattr(chat_models, "vertexai"):
        vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
        vertexai.ChatVertexAI = ChatVertexAI
        chat_models.vertexai = vertexai
        sys.modules["langchain_community.chat_models.vertexai"] = vertexai

    if not hasattr(langchain_community, "llms"):
        llms = types.ModuleType("langchain_community.llms")
        langchain_community.llms = llms
        sys.modules["langchain_community.llms"] = llms
    if not hasattr(langchain_community.llms, "VertexAI"):
        langchain_community.llms.VertexAI = VertexAI
