import warnings
import os
from typing import List, Dict, Any, Optional

import litellm
from litellm import acompletion, aembedding

from logging_setup import setup_logging, get_logger
from types import SimpleNamespace as sn
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


setup_logging()
logger = get_logger(__name__)


os.makedirs("llm_cache", exist_ok=True)
litellm.enable_cache(
    type="disk",
    supported_call_types=["aembedding"],
    disk_cache_dir="llm_cache",
)


models: Dict[str, Any] = {
    "gpt-4o": sn(
        model="gpt-4o",
        meta=sn(
            price_in=2.50,
            price_out=10.00,
            supports_function_calling=True,
        ),
    ),
}


def _compute_function_calling_support_for_models() -> Dict[str, bool]:
    support_map: Dict[str, bool] = {}
    for registry_key, cfg in models.items():
        model_name = getattr(cfg, "model", str(cfg))
        try:
            support_map[registry_key] = bool(
                litellm.supports_function_calling(model=model_name)
            )
        except Exception as error:  # pragma: no cover
            logger.warning(
                f"Function-calling support check failed for '{model_name}': {error}"
            )
            support_map[registry_key] = False
    return support_map


_MODEL_FUNCTION_SUPPORT: Dict[str, bool] = (
    _compute_function_calling_support_for_models()
)
_SUPPORT_SUMMARY_LOGGED = False


def get_model_name(preferred: Optional[str] = None) -> str:
    if preferred and preferred in models:
        return models[preferred].model
    return models["gpt-4o"].model


def extract_token_usage(response) -> Dict[str, int]:
    try:
        if hasattr(response, "usage"):
            usage = response.usage
        elif isinstance(response, dict) and "usage" in response:
            usage = response["usage"]
        else:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if hasattr(usage, "prompt_tokens"):
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
        elif isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
        else:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    except Exception as e:
        logger.warning(f"Failed to extract token usage: {e}")
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def calculate_cost(usage: Dict[str, int], model_key: str) -> Dict[str, float]:
    if model_key not in models:
        return {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}

    model_config = models[model_key]
    if not hasattr(model_config, "meta") or not hasattr(model_config.meta, "price_in"):
        return {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}

    price_per_million_in = model_config.meta.price_in
    price_per_million_out = model_config.meta.price_out

    input_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * price_per_million_in
    output_cost = (
        usage.get("completion_tokens", 0) / 1_000_000
    ) * price_per_million_out
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    return_usage: bool = False,
    **kwargs,
):
    selected = model if model and model not in models else get_model_name(model)

    # Check if function calling is being used
    using_tools = "tools" in kwargs

    global _SUPPORT_SUMMARY_LOGGED
    if not _SUPPORT_SUMMARY_LOGGED:
        for key, supports in _MODEL_FUNCTION_SUPPORT.items():
            logger.info(
                f"Model '{key}' ({models[key].model}) function-calling support: {supports}"
            )
        _SUPPORT_SUMMARY_LOGGED = True

    try:
        selected_supports = bool(litellm.supports_function_calling(model=selected))
    except Exception as error:  # pragma: no cover
        logger.warning(
            f"Could not determine function-calling support for '{selected}': {error}"
        )
        selected_supports = False

    logger.info(
        f"Chat completion using model={selected} | function-calling={selected_supports}"
    )
    response = litellm.completion(model=selected, messages=messages, **kwargs)

    usage = extract_token_usage(response)

    if using_tools:
        if return_usage:
            if hasattr(response, "_usage"):
                response._usage = usage
            else:
                response.usage_info = usage
            return response
        else:
            return response
    else:
        content = (
            response.choices[0].message["content"]
            if hasattr(response.choices[0], "message")
            else response["choices"][0]["message"]["content"]
        )

        if return_usage:
            return content, usage
        else:
            return content


def embed(
    texts: List[str], model: str = "text-embedding-3-small", **kwargs
) -> List[List[float]]:
    logger.info(f"Embedding {len(texts)} texts using model={model}")
    result = litellm.embedding(model=model, input=texts, **kwargs)
    data = result["data"] if isinstance(result, dict) else result.data
    vectors = [
        item["embedding"] if isinstance(item, dict) else item.embedding for item in data
    ]
    return vectors


async def aembed(
    texts: List[str], model: str = "text-embedding-3-small", **kwargs
) -> List[List[float]]:
    logger.info(f"Async embedding {len(texts)} texts using model={model}")
    result = await aembedding(model=model, input=texts, **kwargs)
    data = result["data"] if isinstance(result, dict) else result.data
    vectors = [
        item["embedding"] if isinstance(item, dict) else item.embedding for item in data
    ]
    return vectors


async def achat(messages: List[Dict[str, str]], model: Optional[str] = None, **kwargs):
    selected = model if model and model not in models else get_model_name(model)
    using_tools = "tools" in kwargs
    logger.info(f"Async chat completion using model={selected}")
    response = await acompletion(model=selected, messages=messages, **kwargs)
    if using_tools:
        return response
    content = (
        response.choices[0].message["content"]
        if hasattr(response.choices[0], "message")
        else response["choices"][0]["message"]["content"]
    )
    return content


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LiteLLM helper CLI")
    subparsers = parser.add_subparsers(dest="cmd")

    p_chat = subparsers.add_parser("chat", help="Run a one-off chat completion")
    p_chat.add_argument("prompt", type=str, help="User prompt")
    p_chat.add_argument("--model", type=str, default=None)

    p_emb = subparsers.add_parser("embed", help="Embed text(s)")
    p_emb.add_argument("texts", nargs="+", help="Text strings to embed")
    p_emb.add_argument("--model", type=str, default="text-embedding-3-small")

    args = parser.parse_args()

    if args.cmd == "chat":
        messages = [{"role": "user", "content": args.prompt}]
        output = chat(messages, model=args.model)
        print(output)
    elif args.cmd == "embed":
        vectors = embed(args.texts, model=args.model)
        print(len(vectors), "embeddings; first dim:", len(vectors[0]) if vectors else 0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
