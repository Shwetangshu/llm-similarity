import pandas as pd
from typing import List, Dict

class DataLoaderError(Exception):
    """Custom exception for data loading errors."""
    pass

def load_sentences(file_path: str, gold_col_name: str, translated_col_name: str) -> List[Dict[str, str]]:
    """
    Loads sentences from a CSV file.

    Args:
        file_path: Path to the CSV file.
        gold_col_name: Column name for gold standard sentences.
        translated_col_name: Column name for translated sentences.

    Returns:
        A list of dictionaries, each with 'gold' and 'translated' keys.

    Raises:
        DataLoaderError: If the file is not found or column names are incorrect.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise DataLoaderError(f"Error: File not found at {file_path}")

    if gold_col_name not in df.columns:
        raise DataLoaderError(f"Error: Gold column '{gold_col_name}' not found in {file_path}. Available columns: {df.columns.tolist()}")
    if translated_col_name not in df.columns:
        raise DataLoaderError(f"Error: Translated column '{translated_col_name}' not found in {file_path}. Available columns: {df.columns.tolist()}")

    sentences = []
    for _, row in df.iterrows():
        sentences.append({
            'gold': str(row[gold_col_name]),
            'translated': str(row[translated_col_name])
        })
    return sentences

def save_results(results_data: List[Dict], output_file_path: str) -> None:
    """
    Saves comparison results to a CSV file.

    The input list of dictionaries is expected to have keys like
    'gold_sentence', 'translated_sentence', 'model_A_score',
    'model_B_score', ..., 'average_score'.

    Args:
        results_data: A list of dictionaries containing sentence pairs and scores.
        output_file_path: Path to save the output CSV file.
    """
    if not results_data:
        print("Warning: No results data to save.")
        # Create an empty file or a file with headers if that's preferred
        # For now, just creates an empty df which results in a file with headers
        df = pd.DataFrame([])
    else:
        df = pd.DataFrame(results_data)

    try:
        df.to_csv(output_file_path, index=False)
        print(f"Results saved to {output_file_path}")
    except Exception as e:
        raise DataLoaderError(f"Error saving results to {output_file_path}: {e}")
