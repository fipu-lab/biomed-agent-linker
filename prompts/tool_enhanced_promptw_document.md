# Tool-Enhanced Entity Linking

Task: CUI Entity Linking

For the following candidates:

[[top_10_list]]

Choose the most appropriate CUI (predicted_cui) for the following mention: [[mention]] and context: [[context]]

You must choose from the given candidates list of CUIs [[top_10_list]]. However, if you believe none of the candidates are appropriate matches, you may suggest alternative CUIs outside this list.

Provide your top-1 prediction and rank your alternative_cuis choices. Give brief reasoning (reasoning) for your choice.

You are also given the document [[document]] that the [[mention]] is extracted from.

Use the available tools strategically to investigate and validate candidates. You can call multiple tools, combine their outputs, and use the results to build confidence in your final CUI selection.

## Primary Tools

**`get_candidates(mention, entity_types=None, max_candidates=5)`** - Find CUI candidates (START HERE)
**`get_cui_preferred_term(cui)`** - Verify what a CUI represents - returns prefered medical term
**`get_cui_semantic_types(cui)`** - Check if CUI matches expected entity type

## Advanced Tools (Use When)

**`filter_by_semantic_type(candidates, target_types)`** - Initial candidates are wrong type
**`expand_candidates_with_neighbors(candidates)`** - Need more candidate options
**`get_cui_neighbors(cui)`** - Exploring related concepts
**`find_related_concepts(cui)`** - Deep relationship analysis needed

## Example Strategy

1. **Always start**: `get_candidates(mention, entity_types=[type])`
2. **Check types**: `get_cui_semantic_types()` if candidates seem wrong
3. **If insufficient**: Use `expand_candidates_with_neighbors()` or `filter_by_semantic_type()`
4. **Before deciding**: `get_cui_preferred_term()` to verify final choice

## Example input format

```
For the following candidates:

- C0027051: Myocardial Infarction
- C0155626: Acute myocardial infarction
- C0002962: Angina Pectoris
- C0003473: Aortic Valve Stenosis
- C0027051: Myocardial Infarction

Choose the most appropriate CUI (predicted_cui) for the following mention: myocardial infarction and context: diagnosed with myocardial infarction

You are also given the document [document content would appear here] that the myocardial infarction is extracted from.
```

## Output format

Return ONLY JSON in this exact format:

```json
{
  "predicted_cui": "C1234567",
  "alternative_cuis": [
    "C1234567",
    "C2345678",
    "C3456789",
    "C4567890",
    "C5678901"
  ],
  "reasoning": "explanation for top-1 choice"
}
```
