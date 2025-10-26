#!/usr/bin/env python3

import re
import json

# Test the JSON extraction logic with the actual LLM response
response_text = '```json\n{"predicted_cui": "C0027051", "confidence": 0.85, "reasoning": "The mention \'heart attack\' is a common lay term for \'Myocardial Infarction\'. While \'Acute myocardial infarction\' is a more specific term, the context does not specify the acuity, so the broader term \'Myocardial Infarction\' is more appropriate."}\n```'

print("=== ORIGINAL RESPONSE ===")
print(repr(response_text))

# Step 1: Try to extract JSON from markdown code blocks first
json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
if json_match:
    json_content = json_match.group(1).strip()
    print("\n=== EXTRACTED FROM CODE BLOCK ===")
    print(repr(json_content))
else:
    # Step 2: Try to find JSON-like content in the response
    json_match = re.search(r'\{[^}]*"predicted_cui"[^}]*\}', response_text, re.DOTALL)
    if json_match:
        json_content = json_match.group(0).strip()
        print("\n=== EXTRACTED FROM REGEX ===")
        print(repr(json_content))
    else:
        json_content = response_text.strip()
        print("\n=== USING FULL RESPONSE ===")
        print(repr(json_content))

# Step 3: Clean up common JSON formatting issues
json_content = json_content.replace("\n", " ").replace("\r", " ")
json_content = re.sub(r"\s+", " ", json_content)

print("\n=== CLEANED JSON ===")
print(repr(json_content))

# Step 4: Try to parse
try:
    response_data = json.loads(json_content)
    print("\n=== PARSED SUCCESSFULLY ===")
    print(response_data)
    print(f"predicted_cui: {response_data.get('predicted_cui')}")
except json.JSONDecodeError as e:
    print(f"\n=== JSON PARSING FAILED ===")
    print(f"Error: {e}")
    print(f"Error repr: {repr(str(e))}")
