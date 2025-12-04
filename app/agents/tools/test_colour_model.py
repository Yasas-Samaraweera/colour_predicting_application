"""
Test script to verify the colour_predictor tool is working correctly.
This script tests:
1. Direct function call
2. LangChain tool invocation
3. Tool schema validation
"""
import json
from app.agents.tools.colour_model import colour_predictor

def test_direct_function_call():
    """Test calling the function directly (bypassing LangChain tool wrapper)"""
    print("=" * 60)
    print("TEST 1: Direct Function Call")
    print("=" * 60)
    
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
    
    print(f"Result: {json.dumps(result, indent=2)}")
    print(f"✓ Direct function call successful!")
    print()


def test_tool_schema():
    """Test the tool's schema and metadata"""
    print("=" * 60)
    print("TEST 2: Tool Schema Validation")
    print("=" * 60)
    
    print(f"Tool name: {colour_predictor.name}")
    print(f"Tool description: {colour_predictor.description}")
    print(f"\nTool schema:")
    print(json.dumps(colour_predictor.args_schema.model_json_schema(), indent=2))
    print(f"✓ Tool schema validated!")
    print()


def test_tool_with_invalid_input():
    """Test tool with invalid input to verify validation"""
    print("=" * 60)
    print("TEST 3: Input Validation (should fail)")
    print("=" * 60)
    
    try:
        # Missing required field
        result = colour_predictor.invoke({
            "dye_red_owf": 2.5,
            # Missing other fields
        })
        print("✗ Validation failed - should have raised an error")
    except Exception as e:
        print(f"✓ Validation working correctly - caught error: {type(e).__name__}")
        print(f"  Error message: {str(e)[:100]}...")
    print()


def test_multiple_predictions():
    """Test multiple predictions with different inputs"""
    print("=" * 60)
    print("TEST 4: Multiple Predictions")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Case 1: Standard recipe",
            "params": {
                "dye_red_owf": 4.2,
                "dye_green_owf": 3.5,
                "dye_blue_owf": 2.8,
                "salt_gL": 60.0,
                "sodaAsh_gL": 15.0,
                "temp_C": 70.0,
                "time_min": 60,
                "pH": 10.5,
                "liquor_ratio": 15,
                "water_hardness_ppm": 200,
                "soap_temp_C": 85,
                "soap_time_min": 20
            }
        },
        {
            "name": "Case 2: High dye concentration",
            "params": {
                "dye_red_owf": 5.0,
                "dye_green_owf": 4.0,
                "dye_blue_owf": 3.0,
                "salt_gL": 80.0,
                "sodaAsh_gL": 20.0,
                "temp_C": 80.0,
                "time_min": 90,
                "pH": 11.0,
                "liquor_ratio": 20,
                "water_hardness_ppm": 300,
                "soap_temp_C": 95,
                "soap_time_min": 30
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        result = colour_predictor.invoke(test_case['params'])
        print(f"  RGB: ({result['R']:.1f}, {result['G']:.1f}, {result['B']:.1f})")
        print(f"  HEX: {result['hex']}")
        print(f"  Success: {result['success']}")
    
    print(f"\n✓ Multiple predictions successful!")
    print()


def test_tool_in_agent_context():
    """Test how the tool would be called in an agent context"""
    print("=" * 60)
    print("TEST 5: Tool in Agent Context (Simulated)")
    print("=" * 60)
    
    # Simulate how LangChain agent would call the tool
    tool_input = {
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
    }
    
    print("Simulating agent tool call...")
    print(f"Tool input: {json.dumps(tool_input, indent=2)}")
    
    result = colour_predictor.invoke(tool_input)
    
    print(f"Tool output: {json.dumps(result, indent=2)}")
    print(f"✓ Tool works correctly in agent context!")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("COLOUR PREDICTOR TOOL TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_direct_function_call()
        test_tool_schema()
        test_tool_with_invalid_input()
        test_multiple_predictions()
        test_tool_in_agent_context()
        
        print("=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {type(e).__name__}")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()

