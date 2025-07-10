#!/usr/bin/env python3
"""
Simple test script to verify OpenRouter/Mistral integration.
Run this script to test if the OpenRouter integration is working correctly.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from sdx.agents.client import chat
from sdx.schema.clinical_outputs import LLMDiagnosis


def test_openrouter_integration():
    """Test the OpenRouter integration with a simple medical diagnosis."""
    print("Testing OpenRouter/Mistral integration...")
    
    # Set the API key if not already set
    if not os.environ.get('OPENROUTER_API_KEY'):
        os.environ['OPENROUTER_API_KEY'] = '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'
    
    system_prompt = (
        "You are an experienced physician assistant. "
        "Return a JSON object with keys 'summary' (two sentences) and "
        "'options' (array of differential diagnoses) given the patient data."
    )
    
    user_prompt = """{
        "age": 45,
        "gender": "M",
        "weight_kg": 80,
        "height_cm": 175,
        "symptoms": "chest pain, shortness of breath, fatigue",
        "previous_tests": "none"
    }"""
    
    try:
        print("Sending request to OpenRouter/Mistral...")
        result = chat(system_prompt, user_prompt, session_id="integration_test")
        
        print("\n✅ OpenRouter integration successful!")
        print(f"Summary: {result.summary}")
        print(f"Options: {result.options}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ OpenRouter integration failed: {e}")
        return False


if __name__ == "__main__":
    success = test_openrouter_integration()
    sys.exit(0 if success else 1) 