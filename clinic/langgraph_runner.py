"""Case-local LangGraph loop; environment sources and gold never enter checkpoints."""
from typing import NotRequired, TypedDict
from uuid import uuid4
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class EncounterState(TypedDict):
    evidence: list
    history: list
    turns: int
    rejections: int
    action: dict | None
    final: dict | None
    error: str | None
    working_memory: NotRequired[dict]
    memory_updates: NotRequired[list]


def build_graph(encounter):
    builder = StateGraph(EncounterState)
    builder.add_node("choose_action", encounter.choose)
    builder.add_node("execute_action", encounter.execute)
    builder.add_edge(START, "choose_action")
    builder.add_conditional_edges("choose_action",
        lambda state: END if state["error"] else "execute_action")
    builder.add_conditional_edges("execute_action",
        lambda state: END if state["final"] or state["error"] else "choose_action")
    # shortcut: process-local checkpoints; use SQLite only if crash recovery is required.
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": uuid4().hex},
              "recursion_limit": 2 * (encounter.max_turns + 10) + 5}
    return graph, config
