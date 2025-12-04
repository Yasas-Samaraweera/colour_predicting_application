import json
from typing import Optional

from langchain.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.requirements_agent import requirements_agent  


class RequirementsGraphState(MessagesState):
    requirements_complete: bool
    interruption_message: str
    requirements: Optional[dict]


def requirements_agent_node(state: RequirementsGraphState) -> RequirementsGraphState:
    response = requirements_agent.invoke({"messages": state["messages"]})

    response = response["structured_response"]
    requirements_response = response.requirements

    if requirements_response.missing_info.question != "":
        return {
            "messages": [
                AIMessage(content=requirements_response.missing_info.question)
            ],
            "interruption_message": requirements_response.missing_info.question,
            "requirements_complete": False,
            "requirements": None,
        }

    # Store complete requirements as dict in state
    return {
        "messages": [],
        "requirements_complete": True,
        "interruption_message": "",
        "requirements": requirements_response.model_dump(),
    }


def should_ask_user_for_info(state: RequirementsGraphState) -> bool:
    return not state["requirements_complete"]


def ask_user_for_info(state: RequirementsGraphState) -> RequirementsGraphState:
    user_response = interrupt(state["interruption_message"])

    return {
        "messages": [HumanMessage(content=user_response)],
        "interruption_message": "",
        "requirements_complete": False,
        "requirements": None,
    }


graph = StateGraph(RequirementsGraphState)
graph.add_node("requirements_agent", requirements_agent_node)
graph.add_node("ask_user_for_info", ask_user_for_info)
graph.add_edge(START, "requirements_agent")
graph.add_conditional_edges(
    "requirements_agent",
    should_ask_user_for_info,
    {True: "ask_user_for_info", False: END},
)
graph.add_edge("ask_user_for_info", "requirements_agent")

checkpointer = InMemorySaver()
requirements_graph = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    initial_state = RequirementsGraphState(
        messages=[
            HumanMessage(
                content="I need to check the behaviour of the reactive dyes.red is 4.2%, green is 3.5%, blue is 2.8%."
            )
        ]
    )

    config = {"configurable": {"thread_id": "thread-1"}}

    result = requirements_graph.invoke(initial_state, config)

    while True:
        if "__interrupt__" in result:
            print(result["__interrupt__"])

            user_input = input("")

            current_state = Command(resume=user_input)

            result = requirements_graph.invoke(current_state, config)
        else:
            break

    print(result["requirements"])