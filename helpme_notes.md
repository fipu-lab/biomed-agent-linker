# Code Refactoring Notes

## Overview

The original `main.py` file (2400 lines) has been split into multiple smaller, focused modules for better maintainability and organization.

## New Module Structure

### 1. `main.py` (~200 lines)

- Contains only the configuration and main evaluation function
- Imports functionality from other modules
- Serves as the entry point for the application

### 2. `utils.py`

- General utility functions for file I/O and formatting
- Functions:
  - `get_timestamped_filename`
  - `save_progressive_results`
  - `load_existing_predictions`
  - `should_skip_entity`
  - `load_prompt`
  - `format_candidates_for_prompt`
  - `format_template_with_placeholders`
  - `safe_parse_json`
  - `validate_prediction_in_candidates`
  - `check_predictions_in_candidates`
  - `load_sapbert_candidates_from_json`
  - `get_sapbert_candidates_for_llm`

### 3. `data_loader.py`

- Data loading and candidate generation
- Functions:
  - `load_test_data`
  - `generate_sapbert_candidates_json`

### 4. `tools_handler.py`

- Tool creation and execution for LLM function calling
- Functions:
  - `create_tool_functions`
  - `execute_tool_call`

### 5. `baseline_evaluators.py`

- Evaluation functions for baseline systems
- Functions:
  - `evaluate_baseline` (supports QuickUMLS, SapBERT, BioBERT, SciSpacy)

### 6. `llm_evaluators.py`

- LLM evaluation functions for different configurations
- Functions:
  - `process_single_llm_entity_with_tools`
  - `process_single_llm_entity_no_tools`
  - `evaluate_llm_with_tools_with_document`
  - `evaluate_llm_with_tools_no_document`
  - `evaluate_llm_no_tools_with_document`
  - `evaluate_llm_no_tools_no_document`

### 7. `results_handler.py`

- Results display and saving functions
- Functions:
  - `save_comprehensive_results`
  - `print_comprehensive_comparison`

### 8. `system_initializer.py`

- System initialization for baseline models
- Functions:
  - `initialize_systems`

## Important Notes

### Missing Dependencies

The following modules referenced in the original code appear to be missing:

- `candidate_gen.py` - Contains `get_sapbert_system`, `get_biobert_system` functions
- `scispacy_linker.py` - Contains `get_scispacy_system` function

These imports have been commented out with TODO notes. You'll need to:

1. Create these missing modules, or
2. Find where these functions are actually defined and update the imports accordingly

### Configuration

All configuration remains in `main.py` in the `CONFIG` dictionary, making it easy to adjust settings without diving into implementation details.

### Progressive Saving

The refactored code maintains support for progressive saving of results, which is useful for long-running evaluations that can be resumed if interrupted.
