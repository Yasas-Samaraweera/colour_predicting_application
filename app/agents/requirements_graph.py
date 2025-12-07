import json
from typing import Optional

from langchain.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.requirements_agent import requirements_agent
from app.agents.tools.colour_model import predict_from_requirements


class RequirementsGraphState(MessagesState):
    requirements_complete: bool
    interruption_message: str
    requirements: Optional[dict]
    prediction_result: Optional[dict]


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
            "prediction_result": None,
        }

    # Store complete requirements as dict in state
    return {
        "messages": [],
        "requirements_complete": True,
        "interruption_message": "",
        "requirements": requirements_response.model_dump(),
        "prediction_result": None,
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
        "prediction_result": None,
    }


def make_prediction_node(state: RequirementsGraphState) -> RequirementsGraphState:
    """
    Node that makes RGB predictions using the gathered requirements.
    This node is called after all requirements are collected.
    """
    requirements = state.get("requirements")
    
    if not requirements:
        return {
            "prediction_result": {
                "success": False,
                "error": "No requirements available for prediction.",
            },
        }
    
    print("\n--- Making RGB prediction from gathered requirements ---")
    prediction_result = predict_from_requirements(requirements)
    
    return {
        "prediction_result": prediction_result,
    }


graph = StateGraph(RequirementsGraphState)
graph.add_node("requirements_agent", requirements_agent_node)
graph.add_node("ask_user_for_info", ask_user_for_info)
graph.add_node("make_prediction", make_prediction_node)
graph.add_edge(START, "requirements_agent")
graph.add_conditional_edges(
    "requirements_agent",
    should_ask_user_for_info,
    {True: "ask_user_for_info", False: "make_prediction"},
)
graph.add_edge("ask_user_for_info", "requirements_agent")
graph.add_edge("make_prediction", END)

checkpointer = InMemorySaver()
requirements_graph = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    initial_state = RequirementsGraphState(
        messages=[
            HumanMessage(
                content=""
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

    print("\n=== Gathered Requirements ===")
    print(json.dumps(result["requirements"], indent=2))
    
    if result.get("prediction_result"):
        print("\n=== RGB Prediction Result ===")
        pred = result["prediction_result"]
        if pred.get("success"):
            rgb = pred["predicted_rgb"]
            print(f"Predicted RGB: R={rgb['R']}, G={rgb['G']}, B={rgb['B']}")
            print(f"Hex Color: {pred['hex_color']}")
        else:
            print(f"Prediction failed: {pred.get('error', 'Unknown error')}")