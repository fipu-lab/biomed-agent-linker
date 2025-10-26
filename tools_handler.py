#!/usr/bin/env python3
"""
Tool creation and execution functions for LLM-based entity linking.
"""

from typing import Dict, List, Any
from tools import (
    get_candidates,
    get_cui_preferred_term,
    get_cui_semantic_types,
    filter_by_semantic_type,
    get_cui_neighbors,
    expand_candidates_with_neighbors,
    find_related_concepts,
)


def create_tool_functions() -> List[Dict]:
    """Create tool function definitions for LLM."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_candidates",
                "description": "Find candidate CUIs for a medical entity mention. This tool searches UMLS, filters noise, and optionally filters by semantic type.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mention": {
                            "type": "string",
                            "description": "The medical entity mention to search for (e.g., 'heart attack', 'insulin')",
                        },
                        "entity_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional ST21pv entity types to filter by (e.g., ['DISO'] for diseases, ['CHEM'] for chemicals)",
                        },
                        "max_candidates": {
                            "type": "integer",
                            "description": "Maximum number of candidates to return (default: 5)",
                        },
                    },
                    "required": ["mention"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_cui_preferred_term",
                "description": "Get the canonical/preferred term for a CUI to understand what the concept represents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cui": {
                            "type": "string",
                            "description": "UMLS Concept Unique Identifier (e.g., 'C0027051')",
                        }
                    },
                    "required": ["cui"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_cui_semantic_types",
                "description": "Get semantic types (TUIs) for a CUI to validate its category",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cui": {
                            "type": "string",
                            "description": "UMLS Concept Unique Identifier (e.g., 'C0011860')",
                        }
                    },
                    "required": ["cui"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "filter_by_semantic_type",
                "description": "Filter candidate CUIs by target semantic types (ST21pv).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "candidates": {"type": "array", "items": {"type": "object"}},
                        "target_types": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["candidates", "target_types"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_cui_neighbors",
                "description": "Return related CUIs connected to the given CUI via UMLS relations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cui": {"type": "string"},
                    },
                    "required": ["cui"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "expand_candidates_with_neighbors",
                "description": "Expand candidate set using neighbor relationships. Respects optional type constraints.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "candidates": {"type": "array", "items": {"type": "object"}},
                        "expansion_factor": {
                            "type": "integer",
                            "description": "Neighbors to add per candidate (default 1)",
                        },
                        "additive_bump": {
                            "type": "number",
                            "description": "Score bump for neighbors (default 0.05)",
                        },
                        "target_types": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["candidates"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_related_concepts",
                "description": "Find concepts related to a CUI using graph relations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cui": {"type": "string"},
                        "max_depth": {"type": "integer"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["cui"],
                },
            },
        },
    ]
    return tools


def execute_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Execute a tool call with given arguments."""
    try:
        if tool_name == "get_candidates":
            mention = arguments["mention"]
            entity_types = arguments.get("entity_types")
            max_candidates = arguments.get("max_candidates", 5)
            return get_candidates(mention, entity_types, max_candidates)
        elif tool_name == "get_cui_preferred_term":
            cui = arguments["cui"]
            return get_cui_preferred_term(cui)
        elif tool_name == "get_cui_semantic_types":
            cui = arguments["cui"]
            types = get_cui_semantic_types(cui)
            return list(types)
        elif tool_name == "filter_by_semantic_type":
            candidates = arguments["candidates"]
            target_types = arguments["target_types"]
            return filter_by_semantic_type(candidates, target_types)
        elif tool_name == "get_cui_neighbors":
            cui = arguments["cui"]
            return get_cui_neighbors(cui)
        elif tool_name == "expand_candidates_with_neighbors":
            candidates = arguments["candidates"]
            expansion_factor = arguments.get("expansion_factor", 1)
            additive_bump = arguments.get("additive_bump", 0.05)
            target_types = arguments.get("target_types")
            return expand_candidates_with_neighbors(
                candidates,
                expansion_factor=expansion_factor,
                additive_bump=additive_bump,
                target_types=target_types,
            )
        elif tool_name == "find_related_concepts":
            cui = arguments["cui"]
            max_depth = arguments.get("max_depth", 1)
            max_results = arguments.get("max_results", 10)
            return find_related_concepts(
                cui, max_depth=max_depth, max_results=max_results
            )
        else:
            return f"Unknown tool: {tool_name}"
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"
