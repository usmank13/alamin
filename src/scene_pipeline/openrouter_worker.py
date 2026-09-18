"""Optional SDK worker. No shell tools; stdout is a single JSON envelope."""
import asyncio
import json
from pathlib import Path
import sys
import os

os.environ['PYDANTIC_AI_NO_BANNER']='1'


async def execute(req, model=None):
    from pydantic_ai import Agent, StructuredDict, BinaryContent
    from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
    from pydantic_ai.providers.openrouter import OpenRouterProvider
    from pydantic_ai.usage import UsageLimits
    options = {}
    if req['search']:
        from pydantic_ai.native_tools import WebSearchTool
        from pydantic_ai.capabilities import NativeTool
        options['capabilities'] = [NativeTool(WebSearchTool(max_uses=3))]
    if model is None:model = OpenRouterModel(req['model'], provider=OpenRouterProvider())
    agent = Agent(model, output_type=StructuredDict(req['schema']), retries=0,
                  model_settings=OpenRouterModelSettings(max_tokens=4096,openrouter_usage={'include':True},
                                                        openrouter_provider={'allow_fallbacks': False}), **options)
    prompt = [req['instruction']]
    if req['image']: prompt.append(BinaryContent(data=Path(req['image']).read_bytes(), media_type='image/png'))
    result = await agent.run(prompt,usage_limits=UsageLimits(request_limit=5))
    usage = result.usage
    messages=result.all_messages()
    costs=[(m.provider_details or {}).get('cost') for m in messages if m.kind=='response']
    return dict(output=result.output, usage=dict(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                cost_usd=sum(costs) if costs and all(c is not None for c in costs) else None,
                resolved_model=result.response.model_name,provider_requests=usage.requests),
                messages=json.loads(result.all_messages_json()))


def main():
    from .runtime import redact
    req = json.load(sys.stdin)
    try: print(redact(json.dumps(asyncio.run(execute(req)))))
    except ImportError:
        print(json.dumps({'error': {'code': 'AGENT_DEPENDENCY', 'message': 'Install the agent extra: uv sync --extra pipeline --extra agent'}}))
    except Exception as exc:
        print(json.dumps({'error': {'code': 'AGENT_PROVIDER', 'message': redact(f'{type(exc).__name__}: {exc}')}}))


if __name__ == '__main__': main()
