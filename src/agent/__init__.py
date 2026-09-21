"""LangGraph agents (100% local, sin API): determinista + ReAct.

__init__ perezoso (PEP 562): importar `src.agent.<submodulo>` no debe cargar
el grafo ni las tools, o se crean ciclos (p. ej. graph_pipeline -> progress
-> agent -> tools -> pipeline -> graph_pipeline).
"""

__all__ = [
    "build_agent_graph",
    "build_agent_graph_no_interrupt",
    "build_react_agent",
    "build_react_llm",
    "make_react_tools",
    "AgentState",
]

_LAZY = {
    "build_agent_graph": "src.agent.graph",
    "build_agent_graph_no_interrupt": "src.agent.graph",
    "build_react_agent": "src.agent.react_graph",
    "build_react_llm": "src.agent.react_graph",
    "make_react_tools": "src.agent.tools",
    "AgentState": "src.agent.state",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
