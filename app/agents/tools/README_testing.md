# Testing the Colour Predictor Tool

This guide explains how to verify that the `colour_predictor` tool is being called correctly.

## Quick Test

Run the comprehensive test suite:

```bash
python app/agents/tools/test_colour_model.py
```

This will test:
- Direct function calls
- Tool schema validation
- Input validation
- Multiple predictions
- Agent context simulation

## Manual Testing

### 1. Direct Function Call

```python
from app.agents.tools.colour_model import colour_predictor

result = colour_predictor.invoke({
    "dye_red_owf": 2.5,
    "dye_green_owf": 1.0,
    "dye_blue_owf": 0.5,
    "salt_gL": 50.0,
    "sodaAsh_gL": 15.0,
    "temp_C": 60.8,
    "time_min": 45,
    "pH": 10.5,
    "liquor_ratio": 10,
    "water_hardness_ppm": 150,
    "soap_temp_C": 80,
    "soap_time_min": 15
})

print(result)
# Expected output:
# {
#   "success": True,
#   "R": 116.39,
#   "G": 205.47,
#   "B": 231.37,
#   "hex": "#74cde7"
# }
```

### 2. Check Tool Schema

```python
from app.agents.tools.colour_model import colour_predictor
import json

# View tool metadata
print(f"Tool name: {colour_predictor.name}")
print(f"Tool description: {colour_predictor.description}")

# View input schema
print(json.dumps(colour_predictor.args_schema.model_json_schema(), indent=2))
```

### 3. Enable Logging to Track Tool Calls

The tool includes logging. To see when it's called, enable logging:

```python
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now when the tool is called, you'll see logs like:
# INFO - colour_predictor tool called with parameters: ...
# INFO - colour_predictor prediction: RGB(116.4, 205.5, 231.4) = #74cde7
```

### 4. Test in Agent Context

Test how the agent calls the tool:

```python
from app.agents.requirements_agent import requirements_agent

# The agent will automatically call the tool when needed
response = requirements_agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Predict color for red=4.2%, green=3.5%, blue=2.8%"
    }]
})
```

## Debugging Tips

1. **Check if tool is registered**: Verify the tool is in the agent's tools list
2. **Enable verbose logging**: Set logging level to DEBUG for detailed information
3. **Check tool output**: Verify the return format matches what the agent expects
4. **Validate inputs**: Ensure all 12 required parameters are provided

## Common Issues

- **Missing parameters**: All 12 parameters are required
- **Wrong data types**: Check that integers are int and floats are float
- **Model not loaded**: Ensure `colour_changing_predictor.joblib` exists in `app/ml_models/`
- **Path issues**: The tool uses relative paths, ensure you're running from project root

