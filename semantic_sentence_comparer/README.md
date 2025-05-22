# Semantic Sentence Comparer

This project provides a command-line tool to compare the semantic similarity of sentences from a CSV file. It uses various Natural Language Processing (NLP) models, including Large Language Models (LLMs) like OpenAI's GPT and Google's Gemini, as well as sentence transformer models from Hugging Face. The tool outputs the similarity scores (on a 1-5 scale) and an average score into a new CSV file.

## Features

*   Supports multiple LLM and embedding-based scorers:
    *   OpenAI (e.g., GPT-3.5 Turbo, GPT-4)
    *   Google Gemini (e.g., Gemini-Pro)
    *   Hugging Face Sentence Transformers (e.g., `all-MiniLM-L6-v2`)
*   Scores semantic similarity on a 1-5 scale (1: very different, 5: very similar).
*   Calculates an average score across all selected models for each sentence pair.
*   Input from CSV files, with user-specified columns for gold and translated sentences.
*   Output results to a new CSV file.
*   Configurable via command-line arguments (input/output files, column names, models to use).
*   Extensible design to easily add new scoring models.

## Setup

### 1. Prerequisites

*   Python 3.8 or higher is recommended.

### 2. Clone the Repository (if applicable)

If you haven't downloaded the project files, clone the repository:
```bash
git clone <repository_url>
cd semantic_sentence_comparer
```
(If you already have the files, navigate to the `semantic_sentence_comparer` directory.)

### 3. Create and Activate Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

**For macOS and Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**For Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies

Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

### 5. Set Up API Keys

API keys are required if you plan to use models from OpenAI or Google Gemini.

1.  **Copy the example environment file:**
    The example file `config/.env.example` needs to be copied to the project root directory as `.env`.
    From the `semantic_sentence_comparer` root directory, run:
    ```bash
    cp config/.env.example .env
    ```
    This creates a `.env` file in `semantic_sentence_comparer/.env`.

2.  **Edit the `.env` file:**
    Open the newly created `.env` file with a text editor and fill in your actual API keys:

    ```env
    # .env - Fill in your API keys
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
    # HUGGINGFACE_API_TOKEN="YOUR_HUGGINGFACE_API_TOKEN_HERE" # Optional: Only needed for certain private/gated Hugging Face models
    ```
    Save the file. The application will load these keys automatically.

## Input Data Format

The input data must be provided as a CSV file. This file should contain at least two columns: one for the 'gold standard' (or reference) sentences and one for the 'translated' (or comparison) sentences. You will specify the exact names of these columns when running the script.

**Example CSV structure:**

```csv
id,source_sentence,machine_translation,human_reference
1,"This is an example.","Dies ist ein Beispiel.","This is an example."
2,"Another sentence here.","Ein weiterer Satz hier.","Another sentence is here."
...
```
In this example, you might use `"human_reference"` as your gold column and `"machine_translation"` as your translated column.

## How to Run

The main script for performing comparisons is `main.py`. You run it from the command line within your activated virtual environment.

**Command-line example:**

```bash
python main.py --input_csv data/your_input_data.csv \
               --output_csv data/comparison_results.csv \
               --gold_column "gold_sentence_column_name" \
               --translated_column "translated_sentence_column_name" \
               --scorers openai gemini hf \
               --openai_model "gpt-3.5-turbo" \
               --gemini_model "gemini-pro" \
               --hf_model "sentence-transformers/all-MiniLM-L6-v2" \
               --scorer_weights openai:0.5 gemini:0.3 hf:0.2
```

**Explanation of important arguments:**

*   `--input_csv`: Path to your input CSV file.
*   `--output_csv`: Path where the results CSV file will be saved.
*   `--gold_column`: The name of the column in your input CSV that contains the gold standard sentences.
*   `--translated_column`: The name of the column in your input CSV that contains the sentences to be compared against the gold standard.
*   `--scorers`: A list of scorers to use. Choices are `openai`, `gemini`, `hf`. You can specify one or more (e.g., `--scorers openai hf`). Defaults to all three if not specified.
*   `--openai_model`: (Optional) Specify the OpenAI model to use if `openai` is in `--scorers`. Defaults to "gpt-3.5-turbo".
*   `--gemini_model`: (Optional) Specify the Gemini model to use if `gemini` is in `--scorers`. Defaults to "gemini-pro".
*   `--hf_model`: (Optional) Specify the Hugging Face sentence transformer model to use if `hf` is in `--scorers`. Defaults to "sentence-transformers/all-MiniLM-L6-v2".
*   `--scorer_weights`: (Optional) A list of weights to assign to the scorers, in the format `scorer_name:weight`. For example: `--scorer_weights openai:0.7 gemini:0.3`.
    *   The `scorer_name` must be one of the activated types: `openai`, `gemini`, or `hf`.
    *   Weights must be positive numbers (e.g., 0.5, 1, 2.3).
    *   If this argument is not provided, or if weights are not provided for all active scorers, a simple average of scores will be calculated for each sentence pair.
    *   If `--scorer_weights` are provided, but an active scorer for a specific sentence pair is missing a weight or has a non-positive weight assigned, the script will log a warning and fall back to a simple average for that particular sentence pair. The weights do not need to sum to 1; the calculation will normalize them.

Make sure to replace placeholder paths and column names with your actual data.

## Output Format

The script will generate an output CSV file (specified by `--output_csv`) containing the original sentences along with the similarity scores from each selected model and an average score.

**Example columns in the output CSV:**

*   `gold_sentence`: The original gold standard sentence.
*   `translated_sentence`: The original translated sentence.
*   `openai_gpt-3.5-turbo_score`: Score from the OpenAI model (model name included).
*   `gemini_gemini-pro_score`: Score from the Gemini model (model name included).
*   `hf_sentence-transformers_all-MiniLM-L6-v2_score`: Score from the Hugging Face model (model name included).
*   `average_score`: The average of all collected scores for that sentence pair. If valid `--scorer_weights` were provided and successfully applied for all active scorers for a pair, this will be a weighted average. Otherwise, it will be a simple average. The script logs whether a weighted or simple average was computed for each sentence pair, providing transparency in how the `average_score` was derived.

The exact score column names will depend on the models selected and their names (problematic characters in model names are replaced with underscores).

## Extending with New Scorers

The system is designed to be extensible. To add a new LLM or other scoring mechanism:

1.  **Create a new scorer class:**
    *   In `src/llm_scorers.py`, define a new class that inherits from the abstract base class `LLMScorer`.
    *   Implement the `__init__` method to load any necessary models or API clients.
    *   Implement the `score(self, sentence1: str, sentence2: str) -> float` method. This method should take two sentences as input and return a similarity score (ideally between 1.0 and 5.0).

2.  **Update `main.py`:**
    *   Add an argument to `argparse` in `main.py` if the new scorer needs a specific model name or other parameters.
    *   In the "Initialize Scorers" section of `main.py`, add logic to instantiate your new scorer if it's selected via command-line arguments (e.g., by adding a new choice to the `--scorers` argument).

## Project Structure

```
semantic_sentence_comparer/
├── config/
│   └── .env.example        # Example for API key configuration
├── data/                   # For storing datasets or input/output files (example)
├── src/
│   ├── __init__.py         # Makes 'src' a package
│   ├── data_loader.py      # For loading data and saving results
│   └── llm_scorers.py      # Contains LLMScorer ABC and specific model implementations
├── venv/                   # Python virtual environment (if created as per instructions)
├── .env                    # Actual API key configuration (after copying and editing)
├── main.py                 # Main script to run comparisons
├── README.md               # This file
└── requirements.txt        # Project dependencies
```

## Running Tests

Unit tests are provided to ensure the components of the project are working correctly. The tests use `pytest`.

To run the tests:

1.  Ensure you have installed the development dependencies (including `pytest` and `pytest-mock`), which are part of the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

2.  Navigate to the project root directory (`semantic_sentence_comparer/`) in your terminal.

3.  Run `pytest`:
    ```bash
    pytest
    ```
    Or, for more verbose output:
    ```bash
    pytest -v
    ```

The tests are located in the `tests/` directory and are designed to mock external services like LLM APIs, so they can run without actual API keys or network access during the test execution.
