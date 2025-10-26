#!/usr/bin/env python3

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import chat

# Test simple LLM call
system_prompt = """You are a medical entity linking specialist that maps biomedical text mentions to UMLS CUIs."""

user_prompt = """Select the most appropriate CUI from the provided candidates. You MUST choose from the given candidates only.

**Mention**: "heart attack" | **Type**: DISO | **Context**: "Patient diagnosed with heart attack"
**Candidates**: 
- C0027051: Myocardial Infarction
- C0155626: Acute myocardial infarction

Choose the most appropriate CUI from the candidates above.

Return ONLY JSON in this exact format:
{"predicted_cui": "C1234567", "confidence": 0.95, "reasoning": "explanation"}"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

try:
    response = chat(messages=messages, model="gpt-4o", temperature=0.0, max_tokens=1000)

    print("=== RAW RESPONSE ===")
    print(type(response))
    print(repr(response))
    print("\n=== RESPONSE CONTENT ===")

    if hasattr(response, "content"):
        print(f"response.content: {repr(response.content)}")
    elif hasattr(response, "choices") and response.choices:
        print(
            f"response.choices[0].message.content: {repr(response.choices[0].message.content)}"
        )
    else:
        print(f"str(response): {repr(str(response))}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()
