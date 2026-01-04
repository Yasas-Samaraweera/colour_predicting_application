"""
FastAPI application for Colour Predictor UI.
"""
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agents.requirements_graph import requirements_graph, RequirementsGraphState
from langchain.messages import HumanMessage, AIMessage
from langgraph.types import Command


app = FastAPI(title="Colour Predictor API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (use Redis or database in production)
sessions: Dict[str, dict] = {}


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    requires_input: bool
    prediction_result: Optional[dict] = None
    requirements: Optional[dict] = None


def process_graph_step(session_id: str, user_message: str) -> dict:
    """
    Process a step in the LangGraph workflow.
    """
    config = {"configurable": {"thread_id": session_id}}
    session_exists = session_id in sessions
    
    if not session_exists:
        # Initialize new session
        initial_state = RequirementsGraphState(
            messages=[
                HumanMessage(content=user_message or "")
            ]
        )
        result = requirements_graph.invoke(initial_state, config)
        sessions[session_id] = True  # Mark session as existing
    else:
        # Continue existing session - resume from interruption
        current_state = Command(resume=user_message)
        result = requirements_graph.invoke(current_state, config)
    
    # Check if we need user input (interruption)
    if "__interrupt__" in result:
        # LangGraph returns __interrupt__ as a list of Interrupt objects
        # Each Interrupt object has a "value" attribute
        interrupts = result["__interrupt__"]
        if interrupts and len(interrupts) > 0:
            # Get the interruption message from the first interrupt's value attribute
            interrupt_message = getattr(interrupts[0], "value", "Please provide more information.")
        else:
            interrupt_message = "Please provide more information."
        
        return {
            "requires_input": True,
            "response": interrupt_message,
            "prediction_result": None,
            "requirements": result.get("requirements"),
        }
    
    # Graph completed - extract state from result
    response_text = ""
    
    # Check for prediction result
    if result.get("prediction_result"):
        pred = result["prediction_result"]
        if pred.get("success"):
            rgb = pred["predicted_rgb"]
            response_text = f"✅ Prediction complete! RGB: R={rgb['R']}, G={rgb['G']}, B={rgb['B']} | Hex: {pred['hex_color']}"
        else:
            response_text = f"❌ Prediction failed: {pred.get('error', 'Unknown error')}"
    elif result.get("requirements"):
        # Requirements gathered but prediction not made yet (shouldn't happen with current graph)
        response_text = "Requirements gathered successfully. Processing prediction..."
    
    return {
        "requires_input": False,
        "response": response_text if response_text else "Processing complete.",
        "prediction_result": result.get("prediction_result"),
        "requirements": result.get("requirements"),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Send a message and get a response from the colour predictor agent.
    """
    # Generate or use session ID
    session_id = message.session_id or str(uuid.uuid4())
    
    try:
        result = process_graph_step(session_id, message.message)
        
        return ChatResponse(
            session_id=session_id,
            response=result["response"],
            requires_input=result["requires_input"],
            prediction_result=result["prediction_result"],
            requirements=result["requirements"],
        )
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        error_detail = f"Error processing message: {error_msg}\n{error_traceback}"
        print("=" * 80)
        print("ERROR DETAILS:")
        print(error_detail)
        print("=" * 80)
        raise HTTPException(status_code=500, detail=f"Error: {error_msg}")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/test-graph")
async def test_graph():
    """Test endpoint to check if graph initialization works."""
    try:
        from app.agents.requirements_graph import RequirementsGraphState, requirements_graph
        from langchain.messages import HumanMessage
        
        test_state = RequirementsGraphState(messages=[HumanMessage(content="test")])
        config = {"configurable": {"thread_id": "test-thread"}}
        
        # Just check if we can create the state
        return {
            "status": "ok",
            "state_created": True,
            "requirements_complete": test_state.requirements_complete,
            "interruption_message": test_state.interruption_message
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Get the project root directory (where static/ folder is)
PROJECT_ROOT = Path(__file__).parent.parent.parent
STATIC_DIR = PROJECT_ROOT / "static"


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    html_path = STATIC_DIR / "index.html"
    return FileResponse(str(html_path))


# Mount static files
try:
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
except RuntimeError:
    # Directory doesn't exist yet, will be created
    pass
