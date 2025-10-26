# Standard Entity Linking

Task: CUI Entity Linking

For the following candidates:

[[top_10_list]]

Choose the most appropriate CUI (predicted_cui) for the following mention: [[mention]] and context: [[context]]

You must choose from the given candidates list of CUIs [[top_10_list]]. However, if you believe none of the candidates are appropriate matches, you may suggest alternative CUIs outside this list.

Provide your top-1 prediction and rank your alternative_cuis choices. Give brief reasoning (reasoning) for your choice.

## Example input format

```
For the following candidates:

- C0027051: Myocardial Infarction
- C0155626: Acute myocardial infarction
- C0002962: Angina Pectoris
- C0003473: Aortic Valve Stenosis
- C0027051: Myocardial Infarction

Choose the most appropriate CUI (predicted_cui) for the following mention: myocardial infarction and context: diagnosed with myocardial infarction
```

## Output format

Return only JSON string in this exact format:

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
  "reasoning": "explanation for top-1"
}
```
