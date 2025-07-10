# OpenRouter/Mistral Integration

This document describes the integration of OpenRouter with Mistral AI for medical diagnosis in the TeleHealthCareAI platform.

## Overview

The platform has been updated to use OpenRouter as the primary AI provider, specifically using the Mistral Small 3.2 24B Instruct model for medical diagnosis and treatment recommendations.

## Configuration

### Environment Variables

The following environment variables are used for OpenRouter configuration:

```bash
# OpenRouter Configuration
OPENROUTER_API_KEY=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
OPENROUTER_MODEL=mistralai/mistral-small-3.2-24b-instruct:free
SITE_URL=https://telehealthcareai.com
SITE_NAME=TeleHealthCareAI

# Legacy OpenAI configuration (for backward compatibility)
OPENAI_API_KEY=
OPENAI_MODEL=o4-mini-2025-04-16
```

### Model Details

- **Model**: `mistralai/mistral-small-3.2-24b-instruct:free`
- **Provider**: OpenRouter
- **Base URL**: `https://openrouter.ai/api/v1`
- **Features**: JSON response format, multilingual support

## Implementation Details

### Client Configuration

The main client (`src/sdx/agents/client.py`) has been updated to:

1. Use OpenRouter as the primary API provider
2. Configure Mistral Small 3.2 24B Instruct model
3. Include required headers for OpenRouter analytics
4. Maintain backward compatibility with OpenAI

### Key Changes

1. **Base URL**: Changed from OpenAI to OpenRouter API endpoint
2. **Model**: Switched from `o4-mini-2025-04-16` to `mistralai/mistral-small-3.2-24b-instruct:free`
3. **Headers**: Added OpenRouter-specific headers for analytics
4. **Fallback**: Maintained support for OpenAI API key as fallback

### API Response Format

The integration maintains the same JSON response format:

```json
{
  "summary": "Two-sentence summary of the patient's condition",
  "options": ["diagnosis1", "diagnosis2", "diagnosis3"]
}
```

## Testing

### Running Tests

To test the OpenRouter integration:

```bash
# Run the integration test script
python test_openrouter_integration.py

# Run pytest tests
pytest tests/test_openrouter_client.py -v
```

### Test Coverage

The following tests verify the integration:

1. **Basic Chat Test**: Verifies basic communication with OpenRouter
2. **JSON Response Test**: Ensures proper JSON formatting
3. **Configuration Test**: Validates client setup
4. **Medical Reports Test**: Tests PDF extraction functionality

## Usage Examples

### Basic Diagnosis

```python
from sdx.agents.client import chat

system_prompt = "You are an experienced physician assistant..."
user_prompt = "Patient data in JSON format..."

result = chat(system_prompt, user_prompt, session_id="session_123")
print(f"Summary: {result.summary}")
print(f"Options: {result.options}")
```

### Medical Reports Extraction

```python
from sdx.agents.extraction.medical_reports import get_report_data_from_pdf

fhir_data = get_report_data_from_pdf("medical_report.pdf")
```

## Benefits

1. **Cost Effective**: OpenRouter provides competitive pricing
2. **Model Variety**: Access to multiple AI models through single API
3. **Performance**: Mistral Small 3.2 24B provides excellent medical reasoning
4. **Analytics**: Built-in usage analytics and rankings
5. **Reliability**: Enterprise-grade API infrastructure

## Migration from OpenAI

The integration is designed to be a drop-in replacement for OpenAI:

1. **Same Interface**: All existing code continues to work
2. **Backward Compatibility**: OpenAI API key still supported as fallback
3. **Same Response Format**: No changes to data structures
4. **Enhanced Features**: Better multilingual support and medical reasoning

## Troubleshooting

### Common Issues

1. **API Key Not Found**: Ensure `OPENROUTER_API_KEY` is set
2. **Rate Limiting**: OpenRouter has different rate limits than OpenAI
3. **Model Availability**: Verify model is available in your region

### Debug Mode

Enable debug logging by setting:

```bash
export PYTHONPATH=src
python -c "from sdx.agents.client import chat; print('Client loaded successfully')"
```

## Future Enhancements

1. **Model Selection**: Dynamic model selection based on task
2. **Caching**: Response caching for improved performance
3. **Streaming**: Real-time response streaming
4. **Multi-Modal**: Support for image analysis in medical reports 