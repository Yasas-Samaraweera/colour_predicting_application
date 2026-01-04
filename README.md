# Complete Application Flow - Colour Predictor

This document explains the full end-to-end process of the Colour Predictor application, from when a user sends a message to when they receive a color prediction result.

---

## 🏗️ Architecture Overview

The application uses a **stateful conversational AI workflow** built with:
- **FastAPI**: REST API backend
- **LangGraph**: State machine for workflow orchestration
- **LangChain**: LLM integration and agent framework
- **OpenAI GPT-4**: Natural language understanding
- **Scikit-learn ML Model**: RGB color prediction
- **HTML/CSS/JavaScript**: Frontend chat interface

---

## 📊 High-Level Flow Diagram

```
User Browser → FastAPI Server → LangGraph Workflow → Requirements Agent (LLM) → ML Model → Results → Browser
     ↑                              ↓                          ↓
     └──────────────────────────────┴──────────────────────────┘
                      (Conversation Loop)
```

---

## 🔄 Detailed Step-by-Step Process

### **Phase 1: User Initiates Request**

#### Step 1.1: User Opens Browser
- User navigates to `http://localhost:8000`
- FastAPI serves `static/index.html`
- UI displays: Chat interface with welcome message

#### Step 1.2: User Sends First Message
- User types message in chat (e.g., "I need red=4%, green=3%, blue=2%")
- JavaScript sends POST request to `/api/chat`
- Request body: `{message: "I need red=4%...", session_id: null}`

---

### **Phase 2: FastAPI Receives Request**

#### Step 2.1: Request Handling (`app/api/main.py`)
```python
@app.post("/api/chat")
async def chat(message: ChatMessage):
    session_id = message.session_id or str(uuid.uuid4())  # Generate unique session
    result = process_graph_step(session_id, message.message)
    return ChatResponse(...)
```

#### Step 2.2: Session Management
- If new session: Generate UUID (e.g., "abc-123-def")
- If existing session: Use provided session_id
- Session maintains conversation state via LangGraph checkpointer

#### Step 2.3: Process Graph Step
```python
def process_graph_step(session_id, user_message):
    config = {"configurable": {"thread_id": session_id}}
    
    if not session_exists:
        # New conversation
        initial_state = RequirementsGraphState(messages=[HumanMessage(user_message)])
        result = requirements_graph.invoke(initial_state, config)
    else:
        # Continue conversation (after interruption)
        result = requirements_graph.invoke(Command(resume=user_message), config)
```

---

### **Phase 3: LangGraph Workflow Execution**

The LangGraph workflow is a state machine with these nodes:

```
START → requirements_agent → [Conditional Check]
                              ↓                      ↓
                         ask_user_for_info    make_prediction
                              ↓                      ↓
                         requirements_agent      END
                              ↑
                              └─── (Loop until complete)
```

#### Step 3.1: Initial State
```python
RequirementsGraphState:
  - messages: [HumanMessage("I need red=4%...")]
  - requirements_complete: False
  - requirements: None
  - prediction_result: None
  - interruption_message: ""
```

#### Step 3.2: Enter `requirements_agent` Node

**Location**: `app/agents/requirements_graph.py::requirements_agent_node()`

```python
def requirements_agent_node(state):
    # Invoke the LLM agent
    response = requirements_agent.invoke({"messages": state["messages"]})
    structured_response = response["structured_response"]
    requirements_response = structured_response.requirements
```

**What happens here**:
1. LangChain agent (`requirements_agent`) processes the messages
2. Uses OpenAI GPT-4 to understand user input
3. Extracts structured data using `RequirementsResponseModel` (Pydantic schema)
4. Determines if all required fields are present

#### Step 3.3: Requirements Agent Details (`app/agents/requirements_agent.py`)

**Agent Configuration**:
- **Model**: OpenAI GPT-4 (from `app/core/config.py`)
- **System Prompt**: `requirment_agent_system_prompts` (from `app/agents/prompts/requirements_agent.py`)
- **Tools**: `colour_predictor` tool (available but not used during gathering)
- **Output Format**: `RequirementsResponseModel` (structured output)

**System Prompt Instructions**:
- Analyze user's initial query
- Identify provided information (red, green, blue, salt, soda ash, temperatures, times, etc.)
- Ask targeted questions for missing information
- Validate ranges (e.g., red: 0-5%, salt: 40-80 g/L)

#### Step 3.4: Structured Response Model (`app/agents/response_models/requirements_agent.py`)

The agent returns a `RequirementsResponseModel` containing:

```python
CompleteRequirements:
  - rgb_details: {dye_red_owf, dye_green_owf, dye_blue_owf}
  - chemical_details: {salt_gL, sodaAsh_gL}
  - temperature_details: {temp_C, soap_temp_C}
  - time_details: {time_min, soap_time_min}
  - liquor_ratio: {liquor_ratio}
  - pH: {pH}
  - water_hardness_ppm: {water_hardness_ppm}
  - user_confirmations: {...}
  - missing_info: {missing_fields: [...], question: "..."}
```

#### Step 3.5: Check for Missing Information

```python
if requirements_response.missing_info.question != "":
    # Missing information - need to ask user
    return {
        "requirements_complete": False,
        "interruption_message": requirements_response.missing_info.question,
        ...
    }
else:
    # All information gathered
    return {
        "requirements_complete": True,
        "requirements": requirements_response.model_dump(),
        ...
    }
```

---

### **Phase 4: Conditional Routing**

#### Step 4.1: Conditional Edge (`should_ask_user_for_info`)

```python
def should_ask_user_for_info(state):
    return not state["requirements_complete"]
```

**Routing Decision**:
- If `requirements_complete = False` → Go to `ask_user_for_info` node
- If `requirements_complete = True` → Go to `make_prediction` node

#### Step 4.2a: Path A - Need More Information (`ask_user_for_info` node)

```python
def ask_user_for_info(state):
    user_response = interrupt(state["interruption_message"])
    # This creates an interruption - workflow pauses
    return {
        "messages": [HumanMessage(content=user_response)],
        ...
    }
```

**What happens**:
1. LangGraph **interrupts** the workflow
2. Returns `{"__interrupt__": "question text"}` in the result
3. Workflow waits for user input
4. Graph loops back to `requirements_agent` when user responds

**Example Interruption**:
- Agent asks: "What is the salt concentration in g/L? (Should be between 40-80 g/L)"
- Workflow pauses, sends question to user
- User responds: "50 g/L"
- Workflow resumes from `requirements_agent` node

#### Step 4.2b: Path B - All Information Gathered (`make_prediction` node)

```python
def make_prediction_node(state):
    requirements = state.get("requirements")
    prediction_result = predict_from_requirements(requirements)
    return {"prediction_result": prediction_result}
```

---

### **Phase 5: RGB Prediction (ML Model)**

#### Step 5.1: Extract Requirements (`app/agents/tools/colour_model.py::predict_from_requirements()`)

```python
def predict_from_requirements(requirements):
    rgb_details = requirements["rgb_details"]
    chemical_details = requirements["chemical_details"]
    temperature_details = requirements["temperature_details"]
    time_details = requirements["time_details"]
    # ... extract all fields
```

**Extracts**:
- Red, Green, Blue scales (dye percentages)
- Salt and Soda Ash concentrations
- Dyeing and Soaping temperatures
- Dyeing and Soaping times
- Liquor ratio, pH, Water hardness

#### Step 5.2: Prepare Model Input (`_prepare_model_input()`)

Converts extracted parameters to pandas DataFrame in model format:

```python
model_input = pd.DataFrame({
    "dye_red_owf": [red],
    "dye_green_owf": [green],
    "dye_blue_owf": [blue],
    "salt_gL": [salt],
    "sodaAsh_gL": [soda_ash],
    "temp_C": [dyeing_temperature],
    "time_min": [dyeing_time],
    "pH": [ph_level],
    "liquor_ratio": [liquor_ratio],
    "water_hardness_ppm": [water_hardness],
    "soap_temp_C": [soaping_temperature],
    "soap_time_min": [soaping_time],
})
```

#### Step 5.3: Load ML Model (`_load_model()`)

```python
def _load_model():
    model_path = app/ml_models/colour_changing_predictor.pkl
    model = joblib.load(model_path)  # Load trained scikit-learn model
    return model
```

**Model**: Pre-trained scikit-learn regression model (loaded once, cached globally)

#### Step 5.4: Make Prediction (`_make_prediction()`)

```python
predicted_rgb = model.predict(model_input)  # Returns array: [[R, G, B]]
rgb_r = int(predicted_rgb[0, 0])  # Clamp to 0-255
rgb_g = int(predicted_rgb[0, 1])
rgb_b = int(predicted_rgb[0, 2])
hex_color = f"#{rgb_r:02x}{rgb_g:02x}{rgb_b:02x}"
```

**Returns**:
```python
{
    "success": True,
    "predicted_rgb": {"R": 186, "G": 85, "B": 211},
    "hex_color": "#ba55d3",
    "input_parameters": {...}
}
```

---

### **Phase 6: Return Results to API**

#### Step 6.1: Graph Completes
- LangGraph workflow reaches `END` node
- Result dictionary contains:
  - `requirements`: Complete gathered requirements
  - `prediction_result`: ML model prediction results

#### Step 6.2: Process Graph Result (`process_graph_step()`)

```python
# Check for interruption (shouldn't happen at END)
if "__interrupt__" in result:
    return {"requires_input": True, "response": interrupt_message, ...}

# Extract prediction result
if result.get("prediction_result"):
    pred = result["prediction_result"]
    if pred.get("success"):
        rgb = pred["predicted_rgb"]
        response_text = f"✅ Prediction complete! RGB: R={rgb['R']}, G={rgb['G']}, B={rgb['B']} | Hex: {pred['hex_color']}"
```

#### Step 6.3: Format API Response

```python
return ChatResponse(
    session_id=session_id,
    response="✅ Prediction complete! RGB: R=186, G=85, B=211 | Hex: #ba55d3",
    requires_input=False,
    prediction_result={
        "success": True,
        "predicted_rgb": {"R": 186, "G": 85, "B": 211},
        "hex_color": "#ba55d3"
    },
    requirements={...complete requirements...}
)
```

---

### **Phase 7: Frontend Receives Response**

#### Step 7.1: JavaScript Receives JSON (`static/index.html`)

```javascript
const data = await response.json();
sessionId = data.session_id;
addMessage(data.response, 'assistant', data.prediction_result, data.requirements);
```

#### Step 7.2: Display in UI

1. **Text Response**: Shows success message in chat bubble
2. **Prediction Card**: If `prediction_result.success = true`:
   - Color box showing the predicted color (hex)
   - RGB values display (R, G, B)
   - Hex color code
3. **Requirements Summary**: Shows all gathered parameters

#### Step 7.3: Visual Display

```javascript
function createPredictionCard(predictionResult, requirements) {
    // Creates visual color box with background-color: hex_color
    // Displays RGB values
    // Shows requirements summary
}
```

**User sees**:
- Chat message: "✅ Prediction complete! RGB: R=186, G=85, B=211 | Hex: #ba55d3"
- Color box: Visual representation of the color
- RGB values: R: 186, G: 85, B: 211
- Requirements summary: All parameters used

---

## 🔄 Conversation Loop Example

### Example Conversation Flow:

1. **User**: "I want red=4%, green=3%, blue=2%"
   - Agent extracts: RGB scales ✓
   - Missing: salt, soda ash, temperatures, times, etc.
   - **Agent asks**: "What is the salt concentration in g/L? (40-80 g/L)"
   - **Graph interrupts** → User sees question

2. **User**: "50 g/L"
   - Agent updates: salt ✓
   - Missing: soda ash, temperatures, times, etc.
   - **Agent asks**: "What is the soda ash concentration in g/L? (10-20 g/L)"
   - **Graph interrupts** → User sees question

3. **User**: "15 g/L"
   - ... continues until all parameters gathered ...

4. **User**: "pH is 10.5"
   - Agent checks: All required fields present ✓
   - **Graph routes to**: `make_prediction` node
   - ML model predicts RGB
   - **Response**: Prediction result with color visualization

---

## 🔑 Key Concepts

### **1. State Management**
- Each conversation has a unique `session_id` (UUID)
- LangGraph checkpointer saves state between requests
- State persists: `requirements_complete`, `requirements`, `messages`

### **2. Interruptions**
- LangGraph supports **interruptions** (pausing workflow)
- Used to ask user questions mid-conversation
- Resumes with `Command(resume=user_response)`

### **3. Structured Output**
- LLM returns structured Pydantic models
- Ensures type safety and validation
- `RequirementsResponseModel` enforces schema

### **4. Lazy Model Loading**
- ML model loaded once on first prediction
- Cached in global variable `_loaded_model`
- Avoids reloading on every request

### **5. Session Continuity**
- First request: `session_id = null` → Generate new UUID
- Subsequent requests: Use same `session_id`
- Graph state retrieved from checkpointer

---

## 📁 Key Files and Their Roles

| File | Purpose |
|------|---------|
| `main.py` | Entry point - starts FastAPI server |
| `app/api/main.py` | FastAPI routes and request handling |
| `app/agents/requirements_graph.py` | LangGraph workflow definition |
| `app/agents/requirements_agent.py` | LangChain agent configuration |
| `app/agents/prompts/requirements_agent.py` | System prompt for LLM |
| `app/agents/response_models/requirements_agent.py` | Pydantic schemas for structured output |
| `app/agents/tools/colour_model.py` | ML model loading and prediction |
| `static/index.html` | Frontend UI (HTML/CSS/JavaScript) |
| `app/ml_models/colour_changing_predictor.pkl` | Trained scikit-learn model |
| `app/core/config.py` | Configuration (API keys, model names) |

---

## 🎯 Summary

The application flow is:

1. **User sends message** → FastAPI receives request
2. **FastAPI** → Routes to LangGraph workflow with session management
3. **LangGraph** → Executes state machine:
   - `requirements_agent` node (LLM extracts/gathers info)
   - Conditional check: Complete or ask user?
   - If incomplete → `ask_user_for_info` → Interrupt → Loop back
   - If complete → `make_prediction` → ML model predicts RGB
4. **ML Model** → Loads, predicts, returns RGB + hex color
5. **FastAPI** → Formats response with prediction results
6. **Frontend** → Displays chat message + visual color + requirements summary

The workflow continues in a loop until all requirements are gathered, then makes the final prediction.

---

Maintainer: Yasas Samaraweera  
AI / ML Engineer – Textile & Manufacturing Applications

