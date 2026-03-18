# LLM Configuration

`hiperhealth` diagnostics and exam suggestions use a LiteLLM-backed adapter. You
can change the backend through environment variables or by passing
`LLMSettings(...)` directly.

## Supported provider values

These are the provider names recognized by the library configuration layer:

| Provider value          | Notes                                            | Default model | API key env fallback               |
| ----------------------- | ------------------------------------------------ | ------------- | ---------------------------------- |
| `openai`                | Default provider                                 | `o4-mini`     | `OPENAI_API_KEY`                   |
| `ollama`                | Local/self-hosted via OpenAI-compatible endpoint | `llama3.2:1b` | none                               |
| `cohere`                | Hosted provider                                  | none          | `COHERE_API_KEY`                   |
| `fireworks`             | Hosted provider                                  | none          | `FIREWORKS_API_KEY`                |
| `gemini`                | Google Gemini                                    | none          | `GEMINI_API_KEY`, `GOOGLE_API_KEY` |
| `groq`                  | Hosted provider                                  | none          | `GROQ_API_KEY`                     |
| `huggingface`           | Hosted provider                                  | none          | `HUGGINGFACE_API_KEY`, `HF_TOKEN`  |
| `huggingface-inference` | Hosted provider                                  | none          | `HUGGINGFACE_API_KEY`, `HF_TOKEN`  |
| `together`              | Hosted provider                                  | none          | `TOGETHER_API_KEY`                 |

Compatibility alias:

- `ollama-openai` is accepted and normalized to `ollama`

## Environment variables

Diagnostics use the `HIPERHEALTH_DIAGNOSTICS_LLM_*` namespace:

| Variable                                  | Purpose                                                  |
| ----------------------------------------- | -------------------------------------------------------- |
| `HIPERHEALTH_DIAGNOSTICS_LLM_PROVIDER`    | Provider name. Defaults to `openai`.                     |
| `HIPERHEALTH_DIAGNOSTICS_LLM_MODEL`       | Model name or fully-qualified LiteLLM model string.      |
| `HIPERHEALTH_DIAGNOSTICS_LLM_API_KEY`     | Provider API key.                                        |
| `HIPERHEALTH_DIAGNOSTICS_LLM_BASE_URL`    | Base URL for OpenAI-compatible endpoints such as Ollama. |
| `HIPERHEALTH_DIAGNOSTICS_LLM_TEMPERATURE` | Sampling temperature. Defaults to `0.0`.                 |
| `HIPERHEALTH_DIAGNOSTICS_LLM_MAX_TOKENS`  | Response token limit. Defaults to `800`.                 |
| `HIPERHEALTH_DIAGNOSTICS_LLM_API_PARAMS`  | Extra LiteLLM kwargs as a JSON object.                   |

Generic fallbacks are also supported with `HIPERHEALTH_LLM_*`.

For OpenAI compatibility, these legacy fallbacks are still accepted:

- `OPENAI_MODEL`
- `OPENAI_API_KEY`

Configuration precedence is:

1. `HIPERHEALTH_DIAGNOSTICS_LLM_*`
2. `HIPERHEALTH_LLM_*`
3. OpenAI legacy fallbacks for model and API key
4. Built-in defaults

`HIPERHEALTH_DIAGNOSTICS_LLM_API_PARAMS` must contain a JSON object.

`HIPERHEALTH_DIAGNOSTICS_LLM_BASE_URL` is mapped to LiteLLM's `api_base` unless
`api_base` is already provided inside `HIPERHEALTH_DIAGNOSTICS_LLM_API_PARAMS`.

## Model naming

If the model does not contain a slash, `hiperhealth` builds a LiteLLM model name
as `<provider>/<model>`.

Examples:

- provider `openai` + model `o4-mini` becomes `openai/o4-mini`
- provider `ollama` + model `llama3.2:3b` becomes `ollama/llama3.2:3b`

If `HIPERHEALTH_DIAGNOSTICS_LLM_MODEL` already contains a slash, it is passed
through as-is. This makes other LiteLLM providers possible even if they are not
explicitly listed above.

Important limitation:

- Auto-discovery of API key environment variables only exists for the recognized
  provider names in this page.
- For other LiteLLM providers, set `HIPERHEALTH_DIAGNOSTICS_LLM_API_KEY`
  explicitly.

## Examples

### OpenAI

```bash
export HIPERHEALTH_DIAGNOSTICS_LLM_PROVIDER="openai"
export HIPERHEALTH_DIAGNOSTICS_LLM_MODEL="o4-mini"
export HIPERHEALTH_DIAGNOSTICS_LLM_API_KEY="your-openai-key"
```

### Ollama

```bash
export HIPERHEALTH_DIAGNOSTICS_LLM_PROVIDER="ollama"
export HIPERHEALTH_DIAGNOSTICS_LLM_MODEL="llama3.2:3b"
export HIPERHEALTH_DIAGNOSTICS_LLM_BASE_URL="http://localhost:11434/v1"
```

### Fully-qualified model string

```bash
export HIPERHEALTH_DIAGNOSTICS_LLM_PROVIDER="groq"
export HIPERHEALTH_DIAGNOSTICS_LLM_MODEL="groq/llama-3.3-70b-versatile"
export HIPERHEALTH_DIAGNOSTICS_LLM_API_KEY="your-groq-key"
```

### Passing settings directly in code

```python
from hiperhealth.agents.diagnostics import core as diag
from hiperhealth.llm import LLMSettings

settings = LLMSettings(
    provider='ollama',
    model='llama3.2:3b',
    api_params={'base_url': 'http://localhost:11434/v1'},
    temperature=0.0,
    max_tokens=800,
)

result = diag.differential(
    {'symptoms': 'chest pain'},
    llm_settings=settings,
)
```

## Structured output behavior

The diagnostics adapter asks the selected model to return a JSON object and then
validates it locally with Pydantic.

That means:

- Provider switching is simple
- Validation stays inside `hiperhealth`
- JSON reliability still depends on the selected model/provider
