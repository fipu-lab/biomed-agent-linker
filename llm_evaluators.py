#!/usr/bin/env python3
"""
LLM evaluation functions for the biomedical entity linking system.
"""

import time
import json
from typing import List, Dict, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from logging_setup import get_logger
from llm import chat, models, extract_token_usage, calculate_cost
from utils import (
    load_existing_predictions,
    should_skip_entity,
    save_progressive_results,
    load_prompt,
    format_template_with_placeholders,
    safe_parse_json,
    check_predictions_in_candidates,
)
from tools_handler import create_tool_functions, execute_tool_call


def process_single_llm_entity_with_tools(
    entity: Dict,
    system_prompt: str,
    tools: List[Dict],
    template: str,
    provider_model_name: str,
    candidates_by_mention: Dict[str, List[Dict]],
    config: Dict,
) -> Dict:
    """Process a single entity with LLM tools - designed for concurrent execution."""
    mention = entity["mention"]
    gold_cui = entity["gold_cui"]
    entity_type = entity["types"][0] if entity["types"] else "UNKNOWN"
    context = entity["context"]
    document = entity.get("document", "")

    start_time = time.time()
    tool_calls_made = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if config["verbose"]:
        logger = get_logger(__name__)
        logger.info(f"     Processing mention: '{mention}' (gold: {gold_cui})")

    # Get candidates from pre-generated JSON data
    candidates = candidates_by_mention.get(mention, [])
    if config["verbose"]:
        logger = get_logger(__name__)
        logger.info(f"     Loaded {len(candidates)} candidates for '{mention}'")

    candidate_cuis = [c["cui"] for c in candidates]

    # Use the template with placeholder replacement
    user_prompt = format_template_with_placeholders(
        template, mention, candidates, context, document
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # Initial LLM call with tools
        response = chat(
            messages=messages,
            model=provider_model_name,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            tools=tools,
            tool_choice="auto",
            response_format={"type": "json_object"},
            return_usage=True,
        )

        # Extract usage from response
        if hasattr(response, "usage"):
            usage = extract_token_usage(response)
            for key in total_usage:
                total_usage[key] += usage.get(key, 0)
        elif hasattr(response, "usage_info"):
            usage = response.usage_info
            for key in total_usage:
                total_usage[key] += usage.get(key, 0)

        # Handle tool calling conversation
        conversation_messages = messages.copy()
        max_iterations = 5
        iteration = 0
        # Simple caching / de-dup for tool calls within this entity
        tool_cache: Dict[str, Any] = {}
        seen_calls = set()

        while iteration < max_iterations:
            # Check for tool calls
            tool_calls = None
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = response.tool_calls
            elif (
                hasattr(response, "choices")
                and response.choices
                and hasattr(response.choices[0], "message")
            ):
                message = response.choices[0].message
                if hasattr(message, "tool_calls") and message.tool_calls:
                    tool_calls = message.tool_calls

            if tool_calls:
                # Process tool calls
                content = ""
                if hasattr(response, "content"):
                    content = response.content or ""
                elif hasattr(response, "choices") and response.choices:
                    content = response.choices[0].message.content or ""

                conversation_messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    }
                )

                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    # Create a stable key for caching by tool+args
                    try:
                        cache_key = json.dumps(
                            {"tool": tool_name, "args": arguments}, sort_keys=True
                        )
                    except Exception:
                        cache_key = f"{tool_name}:{str(arguments)}"

                    # Execute tool with caching
                    if cache_key in tool_cache:
                        tool_result = tool_cache[cache_key]
                    else:
                        tool_result = execute_tool_call(tool_name, arguments)
                        tool_cache[cache_key] = tool_result

                    seen_calls.add(cache_key)
                    tool_calls_made.append(
                        {
                            "tool": tool_name,
                            "arguments": arguments,
                            "result": tool_result,
                        }
                    )

                    # Verbose logging for tool calls
                    if config["verbose"]:
                        logger = get_logger(__name__)
                        logger.info(
                            f"     Tool: {tool_name}({arguments}) -> {str(tool_result)[:100]}..."
                        )

                    # Add tool result to conversation
                    conversation_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result),
                        }
                    )

                # Continue conversation
                response = chat(
                    messages=conversation_messages,
                    model=provider_model_name,
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"],
                    tools=tools,
                    tool_choice="auto",
                    response_format={"type": "json_object"},
                    return_usage=True,
                )

                # Extract usage from follow-up response
                if hasattr(response, "usage"):
                    usage = extract_token_usage(response)
                    for key in total_usage:
                        total_usage[key] += usage.get(key, 0)
                elif hasattr(response, "usage_info"):
                    usage = response.usage_info
                    for key in total_usage:
                        total_usage[key] += usage.get(key, 0)
                iteration += 1
            else:
                # No more tool calls, final response
                break

        # Parse final response
        final_content = ""
        if hasattr(response, "content"):
            final_content = response.content or ""
        elif hasattr(response, "choices") and response.choices:
            final_content = response.choices[0].message.content or ""
        else:
            final_content = str(response)

        # Robust JSON parsing
        response_data = safe_parse_json(final_content)
        predicted_cui = response_data.get("predicted_cui")
        alternative_cuis = response_data.get("alternative_cuis", [])

        # Check predictions against candidates (but don't nullify)
        validation_info = check_predictions_in_candidates(
            predicted_cui, alternative_cuis, candidates
        )

    except Exception as e:
        predicted_cui = None
        alternative_cuis = []
        response_data = {"error": str(e)}
        tool_calls_made = []
        validation_info = {"error": "exception_during_validation"}

    end_time = time.time()
    query_time = (end_time - start_time) * 1000

    # Calculate cost (need to find model key from provider_model_name)
    model_key = None
    for key, cfg in models.items():
        if getattr(cfg, "model", str(cfg)) == provider_model_name:
            model_key = key
            break

    cost_info = (
        calculate_cost(total_usage, model_key)
        if model_key
        else {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
    )

    return {
        "mention": mention,
        "gold_cui": gold_cui,
        "predicted_cui": predicted_cui,
        "alternative_cuis": alternative_cuis,
        "query_time_ms": query_time,
        "types": entity["types"],
        "response": response_data,
        "candidates": candidates,
        "candidate_cuis": candidate_cuis,
        "gold_in_candidates": gold_cui in candidate_cuis,
        "validation_info": validation_info,
        "tool_calls": tool_calls_made,
        "approach": "llm_with_tools",
        "pmid": entity.get("pmid"),
        "model": provider_model_name,
        "token_usage": total_usage,
        "cost_info": cost_info,
    }


def process_single_llm_entity_no_tools(
    entity: Dict,
    system_prompt: str,
    template: str,
    provider_model_name: str,
    candidates_by_mention: Dict[str, List[Dict]],
    config: Dict,
) -> Dict:
    """Process a single entity with LLM prompting only - designed for concurrent execution."""
    mention = entity["mention"]
    gold_cui = entity["gold_cui"]
    entity_type = entity["types"][0] if entity["types"] else "UNKNOWN"
    context = entity["context"]
    document = entity.get("document", "")

    # Verbose logging for current entity
    if config["verbose"]:
        logger = get_logger(__name__)
        logger.info(
            f"     Processing mention: '{mention}' (gold: {gold_cui}) [No Tools]"
        )

    # Get candidates from pre-generated JSON data
    candidates = candidates_by_mention.get(mention, [])
    if config["verbose"]:
        logger = get_logger(__name__)
        logger.info(
            f"     Loaded {len(candidates)} candidates for '{mention}' [No Tools]"
        )

    candidate_cuis = [c["cui"] for c in candidates]

    # Use the template with placeholder replacement
    user_prompt = format_template_with_placeholders(
        template, mention, candidates, context, document
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Get LLM response
    start_time = time.time()
    try:
        response, usage = chat(
            messages=messages,
            model=provider_model_name,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            response_format={"type": "json_object"},
            return_usage=True,
        )

        response_text = response
        if hasattr(response, "content"):
            response_text = response.content
        elif hasattr(response, "choices") and response.choices:
            response_text = response.choices[0].message.content
        elif not isinstance(response, str):
            response_text = str(response)
        # Robust JSON parsing
        response_data = safe_parse_json(response_text)
        predicted_cui = response_data.get("predicted_cui")
        alternative_cuis = response_data.get("alternative_cuis", [])

        # Check predictions against candidates (but don't nullify)
        validation_info = check_predictions_in_candidates(
            predicted_cui, alternative_cuis, candidates
        )

    except Exception as e:
        predicted_cui = None
        confidence = 0.0
        response_data = {"error": str(e)}
        alternative_cuis = []
        validation_info = {"error": "exception_during_validation"}
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    end_time = time.time()
    query_time = (end_time - start_time) * 1000

    # Calculate cost (need to find model key from provider_model_name)
    model_key = None
    for key, cfg in models.items():
        if getattr(cfg, "model", str(cfg)) == provider_model_name:
            model_key = key
            break

    cost_info = (
        calculate_cost(usage, model_key)
        if model_key
        else {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
    )

    return {
        "mention": mention,
        "gold_cui": gold_cui,
        "predicted_cui": predicted_cui,
        "alternative_cuis": alternative_cuis,
        "query_time_ms": query_time,
        "types": entity["types"],
        "response": response_data,
        "candidates": candidates,
        "candidate_cuis": candidate_cuis,
        "gold_in_candidates": gold_cui in candidate_cuis,
        "validation_info": validation_info,
        "approach": "llm_no_tools",
        "pmid": entity.get("pmid"),
        "model": provider_model_name,
        "token_usage": usage,
        "cost_info": cost_info,
    }


def evaluate_llm_with_tools_with_document(
    test_entities: List[Dict],
    model_key: str,
    provider_model_name: str,
    candidates_by_mention: Dict[str, List[Dict]],
    config: Dict,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate LLM with function calling enabled and document context using concurrent requests for a specific model."""
    logger = get_logger(__name__)
    logger.info(
        f" Evaluating LLM with function calls enabled and document context... model={model_key} -> {provider_model_name}"
    )

    # Load existing predictions if resume is enabled
    approach_name = f"llm_with_tools_with_document__{model_key}"

    if config.get("progressive_save"):
        existing_predictions, last_pmid = load_existing_predictions(approach_name)
        predictions = existing_predictions.copy()
    else:
        predictions = []
        last_pmid = None

    # Load prompts - use tool-enhanced template with document context
    system_prompt = load_prompt("system_prompt.md")
    template = load_prompt("tool_enhanced_promptw_document.md")
    tools = create_tool_functions()

    correct_at_k = defaultdict(int)
    total_time = 0
    batch_count = 0
    total_tokens_used = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_cost = {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}

    if last_pmid:
        logger.info(f"Resuming from PMID: {last_pmid}")

    # Filter entities to process (for resume functionality)
    entities_to_process = []
    for entity in test_entities:
        if not should_skip_entity(entity, last_pmid, config.get("resume_from_pmid")):
            entities_to_process.append(entity)

    logger.info(
        f" Processing {len(entities_to_process)} entities with {config['max_concurrent_requests']} concurrent requests..."
    )

    # Progress bar
    pbar = tqdm(
        total=len(entities_to_process), desc=f"LLM+Tools:{model_key}", unit="entities"
    )

    # Process entities in batches with concurrent requests
    batch_size = config["max_concurrent_requests"]
    for i in range(0, len(entities_to_process), batch_size):
        batch = entities_to_process[i : i + batch_size]

        # Submit concurrent requests
        with ThreadPoolExecutor(
            max_workers=config["max_concurrent_requests"]
        ) as executor:
            # Submit all tasks in the batch
            future_to_entity = {
                executor.submit(
                    process_single_llm_entity_with_tools,
                    entity,
                    system_prompt,
                    tools,
                    template,
                    provider_model_name,
                    candidates_by_mention,
                    config,
                ): entity
                for entity in batch
            }

            # Collect results as they complete
            batch_results = []
            for future in as_completed(future_to_entity):
                try:
                    result = future.result()
                    batch_results.append(result)

                    # Update metrics
                    predicted_cui = result["predicted_cui"]
                    gold_cui = result["gold_cui"]
                    total_time += result["query_time_ms"]

                    # Aggregate token usage and costs
                    if "token_usage" in result:
                        for key in total_tokens_used:
                            total_tokens_used[key] += result["token_usage"].get(key, 0)
                    if "cost_info" in result:
                        for key in total_cost:
                            total_cost[key] += result["cost_info"].get(key, 0.0)

                    # Check accuracy
                    predicted_list = [predicted_cui] + result.get(
                        "alternative_cuis", []
                    )
                    predicted_list = [cui for cui in predicted_list if cui]

                    for k in config["top_k_values"]:
                        if gold_cui in predicted_list[:k]:
                            correct_at_k[k] += 1

                    # Update progress bar
                    is_correct = "Y" if predicted_cui == gold_cui else "N"
                    mention = result["mention"]
                    pbar.set_postfix(
                        {
                            "Acc@1": f"{correct_at_k[1]/(len(predictions) + len(batch_results))*100:.1f}%",
                            "Last": (
                                f"{mention[:15]}..." if len(mention) > 15 else mention
                            ),
                            "Correct": is_correct,
                        }
                    )
                    pbar.update(1)

                except Exception as e:
                    logger.warning(f"Error processing entity: {e}")
                    entity = future_to_entity[future]
                    # Create error result
                    error_result = {
                        "mention": entity["mention"],
                        "gold_cui": entity["gold_cui"],
                        "predicted_cui": None,
                        "query_time_ms": 0,
                        "types": entity["types"],
                        "response": {"error": str(e)},
                        "tool_calls": [],
                        "approach": "llm_with_tools",
                        "pmid": entity.get("pmid"),
                        "alternative_cuis": [],
                    }
                    batch_results.append(error_result)
                    pbar.update(1)

        # Add batch results to predictions
        predictions.extend(batch_results)

        # Progressive saving
        if (
            config.get("progressive_save")
            and len(predictions) % config["save_batch_size"] == 0
        ):
            batch_count += 1
            if config["verbose"]:
                logger.info(
                    f" Saving progress: {len(predictions)} predictions processed"
                )
            save_progressive_results(approach_name, predictions, batch_num=batch_count)

    pbar.close()

    # Calculate final metrics
    total_entities = len(predictions)
    metrics = {}
    for k in config["top_k_values"]:
        accuracy = correct_at_k[k] / total_entities if total_entities > 0 else 0
        metrics[f"top_{k}_accuracy"] = accuracy

    metrics["avg_time_ms"] = total_time / total_entities if total_entities > 0 else 0
    metrics["queries_per_sec"] = (
        1000 / metrics["avg_time_ms"] if metrics["avg_time_ms"] > 0 else 0
    )
    metrics["total_time_sec"] = total_time / 1000
    metrics["avg_tool_calls"] = (
        sum(len(p["tool_calls"]) for p in predictions) / len(predictions)
        if predictions
        else 0
    )

    # Token usage and cost metrics
    metrics["total_tokens"] = total_tokens_used
    metrics["total_cost"] = total_cost
    metrics["avg_tokens_per_query"] = {
        key: total_tokens_used[key] / total_entities if total_entities > 0 else 0
        for key in total_tokens_used
    }
    metrics["avg_cost_per_query"] = {
        key: total_cost[key] / total_entities if total_entities > 0 else 0.0
        for key in total_cost
    }

    # Final save
    if config.get("progressive_save"):
        save_progressive_results(approach_name, predictions, metrics)

    return metrics, predictions


def evaluate_llm_with_tools_no_document(
    test_entities: List[Dict],
    model_key: str,
    provider_model_name: str,
    candidates_by_mention: Dict[str, List[Dict]],
    config: Dict,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate LLM with function calling enabled but no document context using concurrent requests for a specific model."""
    logger = get_logger(__name__)
    logger.info(
        f" Evaluating LLM with function calls enabled but no document context... model={model_key} -> {provider_model_name}"
    )

    # Load existing predictions if resume is enabled
    approach_name = f"llm_with_tools_no_document__{model_key}"

    if config.get("progressive_save"):
        existing_predictions, last_pmid = load_existing_predictions(approach_name)
        predictions = existing_predictions.copy()
    else:
        predictions = []
        last_pmid = None

    # Load prompts - use tool-enhanced template without document context
    system_prompt = load_prompt("system_prompt.md")
    template = load_prompt("tool_enhanced_prompt.md")
    tools = create_tool_functions()

    correct_at_k = defaultdict(int)
    total_time = 0
    batch_count = 0
    total_tokens_used = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_cost = {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}

    if last_pmid:
        logger.info(f"Resuming from PMID: {last_pmid}")

    # Filter entities to process (for resume functionality)
    entities_to_process = []
    for entity in test_entities:
        if not should_skip_entity(entity, last_pmid, config.get("resume_from_pmid")):
            entities_to_process.append(entity)

    logger.info(
        f" Processing {len(entities_to_process)} entities with {config['max_concurrent_requests']} concurrent requests..."
    )

    # Progress bar
    pbar = tqdm(
        total=len(entities_to_process),
        desc=f"LLM+Tools-NoDoc:{model_key}",
        unit="entities",
    )

    # Process entities in batches with concurrent requests
    batch_size = config["max_concurrent_requests"]
    for i in range(0, len(entities_to_process), batch_size):
        batch = entities_to_process[i : i + batch_size]

        # Submit concurrent requests
        with ThreadPoolExecutor(
            max_workers=config["max_concurrent_requests"]
        ) as executor:
            # Submit all tasks in the batch
            future_to_entity = {
                executor.submit(
                    process_single_llm_entity_with_tools,
                    entity,
                    system_prompt,
                    tools,
                    template,
                    provider_model_name,
                    candidates_by_mention,
                    config,
                ): entity
                for entity in batch
            }

            # Collect results as they complete
            batch_results = []
            for future in as_completed(future_to_entity):
                try:
                    result = future.result()
                    batch_results.append(result)

                    # Update metrics
                    predicted_cui = result["predicted_cui"]
                    gold_cui = result["gold_cui"]
                    total_time += result["query_time_ms"]

                    # Aggregate token usage and costs
                    if "token_usage" in result:
                        for key in total_tokens_used:
                            total_tokens_used[key] += result["token_usage"].get(key, 0)
                    if "cost_info" in result:
                        for key in total_cost:
                            total_cost[key] += result["cost_info"].get(key, 0.0)

                    # Check accuracy
                    predicted_list = [predicted_cui] + result.get(
                        "alternative_cuis", []
                    )
                    predicted_list = [cui for cui in predicted_list if cui]

                    for k in config["top_k_values"]:
                        if gold_cui in predicted_list[:k]:
                            correct_at_k[k] += 1

                    # Update progress bar
                    is_correct = "Y" if predicted_cui == gold_cui else "N"
                    mention = result["mention"]
                    pbar.set_postfix(
                        {
                            "Acc@1": f"{correct_at_k[1]/(len(predictions) + len(batch_results))*100:.1f}%",
                            "Last": (
                                f"{mention[:15]}..." if len(mention) > 15 else mention
                            ),
                            "Correct": is_correct,
                        }
                    )
                    pbar.update(1)

                except Exception as e:
                    logger.warning(f"Error processing entity: {e}")
                    entity = future_to_entity[future]
                    # Create error result
                    error_result = {
                        "mention": entity["mention"],
                        "gold_cui": entity["gold_cui"],
                        "predicted_cui": None,
                        "query_time_ms": 0,
                        "types": entity["types"],
                        "response": {"error": str(e)},
                        "tool_calls": [],
                        "approach": "llm_with_tools_no_document",
                        "pmid": entity.get("pmid"),
                        "alternative_cuis": [],
                    }
                    batch_results.append(error_result)
                    pbar.update(1)

        # Add batch results to predictions
        predictions.extend(batch_results)

        # Progressive saving
        if (
            config.get("progressive_save")
            and len(predictions) % config["save_batch_size"] == 0
        ):
            batch_count += 1
            if config["verbose"]:
                logger.info(
                    f" Saving progress: {len(predictions)} predictions processed"
                )
            save_progressive_results(approach_name, predictions, batch_num=batch_count)

    pbar.close()

    # Calculate final metrics
    total_entities = len(predictions)
    metrics = {}
    for k in config["top_k_values"]:
        accuracy = correct_at_k[k] / total_entities if total_entities > 0 else 0
        metrics[f"top_{k}_accuracy"] = accuracy

    metrics["avg_time_ms"] = total_time / total_entities if total_entities > 0 else 0
    metrics["queries_per_sec"] = (
        1000 / metrics["avg_time_ms"] if metrics["avg_time_ms"] > 0 else 0
    )
    metrics["total_time_sec"] = total_time / 1000
    metrics["avg_tool_calls"] = (
        sum(len(p["tool_calls"]) for p in predictions) / len(predictions)
        if predictions
        else 0
    )

    # Token usage and cost metrics
    metrics["total_tokens"] = total_tokens_used
    metrics["total_cost"] = total_cost
    metrics["avg_tokens_per_query"] = {
        key: total_tokens_used[key] / total_entities if total_entities > 0 else 0
        for key in total_tokens_used
    }
    metrics["avg_cost_per_query"] = {
        key: total_cost[key] / total_entities if total_entities > 0 else 0.0
        for key in total_cost
    }

    # Final save
    if config.get("progressive_save"):
        save_progressive_results(approach_name, predictions, metrics)

    return metrics, predictions


def evaluate_llm_no_tools_with_document(
    test_entities: List[Dict],
    model_key: str,
    provider_model_name: str,
    candidates_by_mention: Dict[str, List[Dict]],
    config: Dict,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate LLM with prompting only and document context using concurrent requests for a specific model."""
    logger = get_logger(__name__)
    logger.info(
        f" Evaluating LLM with prompting only and document context... model={model_key} -> {provider_model_name}"
    )

    # Load existing predictions if resume is enabled
    approach_name = f"llm_no_tools_with_document__{model_key}"

    if config.get("progressive_save"):
        existing_predictions, last_pmid = load_existing_predictions(approach_name)
        predictions = existing_predictions.copy()
    else:
        predictions = []
        last_pmid = None

    # Load prompts - use standard template with document context
    system_prompt = load_prompt("system_prompt.md")
    template = load_prompt("standard_linking_prompt_w_document.md")

    correct_at_k = defaultdict(int)
    total_time = 0
    batch_count = 0
    total_tokens_used = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_cost = {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}

    if last_pmid:
        logger.info(f"Resuming from PMID: {last_pmid}")

    # Filter entities to process (for resume functionality)
    entities_to_process = []
    for entity in test_entities:
        if not should_skip_entity(entity, last_pmid, config.get("resume_from_pmid")):
            entities_to_process.append(entity)

    logger.info(
        f" Processing {len(entities_to_process)} entities with {config['max_concurrent_requests']} concurrent requests..."
    )

    # Progress bar
    pbar = tqdm(
        total=len(entities_to_process),
        desc=f"LLM-Only-WithDoc:{model_key}",
        unit="entities",
    )

    # Process entities in batches with concurrent requests
    batch_size = config["max_concurrent_requests"]
    for i in range(0, len(entities_to_process), batch_size):
        batch = entities_to_process[i : i + batch_size]

        # Submit concurrent requests
        with ThreadPoolExecutor(
            max_workers=config["max_concurrent_requests"]
        ) as executor:
            # Submit all tasks in the batch
            future_to_entity = {
                executor.submit(
                    process_single_llm_entity_no_tools,
                    entity,
                    system_prompt,
                    template,
                    provider_model_name,
                    candidates_by_mention,
                    config,
                ): entity
                for entity in batch
            }

            # Collect results as they complete
            batch_results = []
            for future in as_completed(future_to_entity):
                try:
                    result = future.result()
                    batch_results.append(result)

                    # Update metrics
                    predicted_cui = result["predicted_cui"]
                    gold_cui = result["gold_cui"]
                    total_time += result["query_time_ms"]

                    # Aggregate token usage and costs
                    if "token_usage" in result:
                        for key in total_tokens_used:
                            total_tokens_used[key] += result["token_usage"].get(key, 0)
                    if "cost_info" in result:
                        for key in total_cost:
                            total_cost[key] += result["cost_info"].get(key, 0.0)

                    # Check accuracy
                    predicted_list = [predicted_cui] + result.get(
                        "alternative_cuis", []
                    )
                    predicted_list = [cui for cui in predicted_list if cui]

                    for k in config["top_k_values"]:
                        if gold_cui in predicted_list[:k]:
                            correct_at_k[k] += 1

                    # Update progress bar
                    is_correct = "Y" if predicted_cui == gold_cui else "N"
                    mention = result["mention"]
                    pbar.set_postfix(
                        {
                            "Acc@1": f"{correct_at_k[1]/(len(predictions) + len(batch_results))*100:.1f}%",
                            "Last": (
                                f"{mention[:15]}..." if len(mention) > 15 else mention
                            ),
                            "Correct": is_correct,
                        }
                    )
                    pbar.update(1)

                except Exception as e:
                    logger.warning(f"Error processing entity: {e}")
                    entity = future_to_entity[future]
                    # Create error result
                    error_result = {
                        "mention": entity["mention"],
                        "gold_cui": entity["gold_cui"],
                        "predicted_cui": None,
                        "query_time_ms": 0,
                        "types": entity["types"],
                        "response": {"error": str(e)},
                        "candidates": [],
                        "candidate_cuis": [],
                        "gold_in_candidates": False,
                        "approach": "llm_no_tools_with_document",
                        "pmid": entity.get("pmid"),
                        "alternative_cuis": [],
                    }
                    batch_results.append(error_result)
                    pbar.update(1)

        # Add batch results to predictions
        predictions.extend(batch_results)

        # Progressive saving
        if (
            config.get("progressive_save")
            and len(predictions) % config["save_batch_size"] == 0
        ):
            batch_count += 1
            if config["verbose"]:
                logger.info(
                    f" Saving progress: {len(predictions)} predictions processed"
                )
            save_progressive_results(approach_name, predictions, batch_num=batch_count)

    pbar.close()

    # Calculate final metrics
    total_entities = len(predictions)
    metrics = {}
    for k in config["top_k_values"]:
        accuracy = correct_at_k[k] / total_entities if total_entities > 0 else 0
        metrics[f"top_{k}_accuracy"] = accuracy

    metrics["avg_time_ms"] = total_time / total_entities if total_entities > 0 else 0
    metrics["queries_per_sec"] = (
        1000 / metrics["avg_time_ms"] if metrics["avg_time_ms"] > 0 else 0
    )
    metrics["total_time_sec"] = total_time / 1000

    # Token usage and cost metrics
    metrics["total_tokens"] = total_tokens_used
    metrics["total_cost"] = total_cost
    metrics["avg_tokens_per_query"] = {
        key: total_tokens_used[key] / total_entities if total_entities > 0 else 0
        for key in total_tokens_used
    }
    metrics["avg_cost_per_query"] = {
        key: total_cost[key] / total_entities if total_entities > 0 else 0.0
        for key in total_cost
    }

    # Final save
    if config.get("progressive_save"):
        save_progressive_results(approach_name, predictions, metrics)

    return metrics, predictions


def evaluate_llm_no_tools_no_document(
    test_entities: List[Dict],
    model_key: str,
    provider_model_name: str,
    candidates_by_mention: Dict[str, List[Dict]],
    config: Dict,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate LLM with prompting only and no document context using concurrent requests for a specific model."""
    logger = get_logger(__name__)
    logger.info(
        f" Evaluating LLM with prompting only and no document context... model={model_key} -> {provider_model_name}"
    )

    # Load existing predictions if resume is enabled
    approach_name = f"llm_no_tools_no_document__{model_key}"

    if config.get("progressive_save"):
        existing_predictions, last_pmid = load_existing_predictions(approach_name)
        predictions = existing_predictions.copy()
    else:
        predictions = []
        last_pmid = None

    # Load prompts - use standard template without document context
    system_prompt = load_prompt("system_prompt.md")
    template = load_prompt("standard_linking_prompt.md")

    correct_at_k = defaultdict(int)
    total_time = 0
    batch_count = 0
    total_tokens_used = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_cost = {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}

    if last_pmid:
        logger.info(f"Resuming from PMID: {last_pmid}")

    # Filter entities to process (for resume functionality)
    entities_to_process = []
    for entity in test_entities:
        if not should_skip_entity(entity, last_pmid, config.get("resume_from_pmid")):
            entities_to_process.append(entity)

    logger.info(
        f" Processing {len(entities_to_process)} entities with {config['max_concurrent_requests']} concurrent requests..."
    )

    # Progress bar
    pbar = tqdm(
        total=len(entities_to_process),
        desc=f"LLM-Only-NoDoc:{model_key}",
        unit="entities",
    )

    # Process entities in batches with concurrent requests
    batch_size = config["max_concurrent_requests"]
    for i in range(0, len(entities_to_process), batch_size):
        batch = entities_to_process[i : i + batch_size]

        # Submit concurrent requests
        with ThreadPoolExecutor(
            max_workers=config["max_concurrent_requests"]
        ) as executor:
            # Submit all tasks in the batch
            future_to_entity = {
                executor.submit(
                    process_single_llm_entity_no_tools,
                    entity,
                    system_prompt,
                    template,
                    provider_model_name,
                    candidates_by_mention,
                    config,
                ): entity
                for entity in batch
            }

            # Collect results as they complete
            batch_results = []
            for future in as_completed(future_to_entity):
                try:
                    result = future.result()
                    batch_results.append(result)

                    # Update metrics
                    predicted_cui = result["predicted_cui"]
                    gold_cui = result["gold_cui"]
                    total_time += result["query_time_ms"]

                    # Aggregate token usage and costs
                    if "token_usage" in result:
                        for key in total_tokens_used:
                            total_tokens_used[key] += result["token_usage"].get(key, 0)
                    if "cost_info" in result:
                        for key in total_cost:
                            total_cost[key] += result["cost_info"].get(key, 0.0)

                    # Check accuracy
                    predicted_list = [predicted_cui] + result.get(
                        "alternative_cuis", []
                    )
                    predicted_list = [cui for cui in predicted_list if cui]

                    for k in config["top_k_values"]:
                        if gold_cui in predicted_list[:k]:
                            correct_at_k[k] += 1

                    # Update progress bar
                    is_correct = "Y" if predicted_cui == gold_cui else "N"
                    mention = result["mention"]
                    pbar.set_postfix(
                        {
                            "Acc@1": f"{correct_at_k[1]/(len(predictions) + len(batch_results))*100:.1f}%",
                            "Last": (
                                f"{mention[:15]}..." if len(mention) > 15 else mention
                            ),
                            "Correct": is_correct,
                        }
                    )
                    pbar.update(1)

                except Exception as e:
                    logger.warning(f"Error processing entity: {e}")
                    entity = future_to_entity[future]
                    # Create error result
                    error_result = {
                        "mention": entity["mention"],
                        "gold_cui": entity["gold_cui"],
                        "predicted_cui": None,
                        "query_time_ms": 0,
                        "types": entity["types"],
                        "response": {"error": str(e)},
                        "candidates": [],
                        "candidate_cuis": [],
                        "gold_in_candidates": False,
                        "approach": "llm_no_tools_no_document",
                        "pmid": entity.get("pmid"),
                        "alternative_cuis": [],
                    }
                    batch_results.append(error_result)
                    pbar.update(1)

        # Add batch results to predictions
        predictions.extend(batch_results)

        # Progressive saving
        if (
            config.get("progressive_save")
            and len(predictions) % config["save_batch_size"] == 0
        ):
            batch_count += 1
            if config["verbose"]:
                logger.info(
                    f" Saving progress: {len(predictions)} predictions processed"
                )
            save_progressive_results(approach_name, predictions, batch_num=batch_count)

    pbar.close()

    # Calculate final metrics
    total_entities = len(predictions)
    metrics = {}
    for k in config["top_k_values"]:
        accuracy = correct_at_k[k] / total_entities if total_entities > 0 else 0
        metrics[f"top_{k}_accuracy"] = accuracy

    metrics["avg_time_ms"] = total_time / total_entities if total_entities > 0 else 0
    metrics["queries_per_sec"] = (
        1000 / metrics["avg_time_ms"] if metrics["avg_time_ms"] > 0 else 0
    )
    metrics["total_time_sec"] = total_time / 1000

    # Token usage and cost metrics
    metrics["total_tokens"] = total_tokens_used
    metrics["total_cost"] = total_cost
    metrics["avg_tokens_per_query"] = {
        key: total_tokens_used[key] / total_entities if total_entities > 0 else 0
        for key in total_tokens_used
    }
    metrics["avg_cost_per_query"] = {
        key: total_cost[key] / total_entities if total_entities > 0 else 0.0
        for key in total_cost
    }

    # Final save
    if config.get("progressive_save"):
        save_progressive_results(approach_name, predictions, metrics)

    return metrics, predictions
