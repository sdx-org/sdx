"""Tests for OpenRouter/Mistral client integration."""

import os
import pytest

from sdx.agents.client import chat
from sdx.schema.clinical_outputs import LLMDiagnosis


@pytest.mark.skipif(
    not os.environ.get('OPENROUTER_API_KEY'), 
    reason='OpenRouter API key not available'
)
def test_openrouter_chat_basic():
    """Test basic chat functionality with OpenRouter/Mistral."""
    system_prompt = "You are a helpful medical assistant. Return a JSON object with 'summary' and 'options' fields."
    user_prompt = "Patient has a headache and fever. What could this be?"
    
    try:
        result = chat(system_prompt, user_prompt, session_id="test_session")
        
        # Verify the result is a valid LLMDiagnosis
        assert isinstance(result, LLMDiagnosis)
        assert hasattr(result, 'summary')
        assert hasattr(result, 'options')
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0
        assert isinstance(result.options, (list, dict))
        
    except Exception as e:
        pytest.fail(f"OpenRouter chat test failed: {e}")


@pytest.mark.skipif(
    not os.environ.get('OPENROUTER_API_KEY'), 
    reason='OpenRouter API key not available'
)
def test_openrouter_json_response():
    """Test that OpenRouter returns valid JSON responses."""
    system_prompt = (
        "You are a medical assistant. Return ONLY a JSON object with these exact keys: "
        "'summary' (string) and 'options' (array of strings)."
    )
    user_prompt = "Patient has chest pain. List 3 possible causes."
    
    try:
        result = chat(system_prompt, user_prompt, session_id="test_json")
        
        # Verify JSON structure
        assert isinstance(result, LLMDiagnosis)
        assert isinstance(result.summary, str)
        assert isinstance(result.options, (list, dict))
        
        # If options is a list, verify it contains strings
        if isinstance(result.options, list):
            assert all(isinstance(option, str) for option in result.options)
            
    except Exception as e:
        pytest.fail(f"OpenRouter JSON response test failed: {e}")


def test_openrouter_configuration():
    """Test that OpenRouter configuration is properly set up."""
    # This test doesn't require an API call, just checks configuration
    from sdx.agents.client import _client, _MODEL_NAME, _OPENROUTER_API_KEY
    
    assert _client.base_url == "https://openrouter.ai/api/v1"
    assert _MODEL_NAME == "mistralai/mistral-small-3.2-24b-instruct:free"
    assert _OPENROUTER_API_KEY is not None
    assert len(_OPENROUTER_API_KEY) > 0 