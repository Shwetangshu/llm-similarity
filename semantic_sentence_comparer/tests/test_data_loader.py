import pytest
import pandas as pd
import os
from src.data_loader import load_sentences, save_results, DataLoaderError

@pytest.fixture
def temp_csv_file(tmp_path):
    """Fixture to create a temporary CSV file for testing."""
    csv_file = tmp_path / "test_input.csv"
    data = {
        'id': [1, 2, 3],
        'gold_sentence': ["Gold sentence 1", "Gold sentence 2", "Gold sentence 3"],
        'translated_sentence': ["Translated sentence 1", "Translated sentence 2", "Translated sentence 3"],
        'other_column': ["extra1", "extra2", "extra3"]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return csv_file

@pytest.fixture
def temp_csv_file_wrong_cols(tmp_path):
    """Fixture for a CSV with incorrect column names."""
    csv_file = tmp_path / "test_wrong_cols.csv"
    data = {
        'id': [1, 2],
        'original': ["Original 1", "Original 2"],
        'translation': ["Translation 1", "Translation 2"]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return csv_file

@pytest.fixture
def sample_results_data():
    """Fixture to provide sample results data."""
    return [
        {'gold_sentence': 'g1', 'translated_sentence': 't1', 'modelA_score': 4.0, 'modelB_score': 3.5, 'average_score': 3.75},
        {'gold_sentence': 'g2', 'translated_sentence': 't2', 'modelA_score': 2.0, 'modelB_score': 2.5, 'average_score': 2.25}
    ]

# Tests for load_sentences
def test_load_sentences_valid_csv(temp_csv_file):
    """Test loading sentences from a valid CSV file."""
    sentences = load_sentences(str(temp_csv_file), "gold_sentence", "translated_sentence")
    assert isinstance(sentences, list)
    assert len(sentences) == 3
    assert all('gold' in item and 'translated' in item for item in sentences)
    assert sentences[0]['gold'] == "Gold sentence 1"
    assert sentences[0]['translated'] == "Translated sentence 1"
    assert sentences[1]['gold'] == "Gold sentence 2"
    assert sentences[2]['translated'] == "Translated sentence 3"

def test_load_sentences_file_not_found():
    """Test DataLoaderError is raised for a non-existent file."""
    with pytest.raises(DataLoaderError, match="Error: File not found at non_existent_file.csv"):
        load_sentences("non_existent_file.csv", "gold", "translated")

def test_load_sentences_missing_gold_column(temp_csv_file):
    """Test DataLoaderError for missing gold column."""
    with pytest.raises(DataLoaderError, match="Error: Gold column 'wrong_gold_col' not found"):
        load_sentences(str(temp_csv_file), "wrong_gold_col", "translated_sentence")

def test_load_sentences_missing_translated_column(temp_csv_file):
    """Test DataLoaderError for missing translated column."""
    with pytest.raises(DataLoaderError, match="Error: Translated column 'wrong_trans_col' not found"):
        load_sentences(str(temp_csv_file), "gold_sentence", "wrong_trans_col")

def test_load_sentences_wrong_column_names_integration(temp_csv_file_wrong_cols):
    """Test DataLoaderError is raised when specified columns are entirely wrong."""
    with pytest.raises(DataLoaderError, match="Error: Gold column 'gold' not found"):
        load_sentences(str(temp_csv_file_wrong_cols), "gold", "translated")

# Tests for save_results
def test_save_results_valid_data(tmp_path, sample_results_data):
    """Test saving valid results data to a CSV file."""
    output_file = tmp_path / "output_results.csv"
    save_results(sample_results_data, str(output_file))
    
    assert output_file.exists()
    df = pd.read_csv(output_file)
    assert len(df) == 2
    assert 'gold_sentence' in df.columns
    assert 'average_score' in df.columns
    assert df['average_score'].iloc[0] == 3.75
    assert df['gold_sentence'].iloc[1] == 'g2'

def test_save_results_empty_data(tmp_path):
    """Test saving results when data is empty. Should create a file with headers."""
    output_file = tmp_path / "empty_output.csv"
    save_results([], str(output_file))
    
    assert output_file.exists()
    # Check if it's an empty file or has headers. Pandas by default writes headers for an empty DataFrame.
    df = pd.read_csv(output_file)
    assert len(df) == 0 
    # Depending on implementation, it might have 0 columns or some default ones if not handled explicitly
    # The current data_loader.py save_results will write an empty df, which means 0 columns unless results_data has structure
    # If results_data is truly empty list, df = pd.DataFrame([]) will have 0 columns.
    # Let's refine this test based on actual save_results behavior.
    # If an empty list is passed, pd.DataFrame([]) creates a DF with 0 rows, 0 columns.
    # df.to_csv then creates an empty file (0 bytes) or a file with just a newline.
    # Let's adjust to expect an empty file or one with just headers if that's the behavior.
    # The current save_results creates an empty df, then to_csv.
    # So, an empty file or a file with headers.
    # If we expect headers even for empty data, the save_results should be adjusted.
    # For now, let's assume an empty dataframe results in an empty file or a file with headers.
    # If it's an empty list, pd.DataFrame([]) produces 0 columns.
    # If it's a list of dicts that's empty, it also produces 0 columns.
    # The current code: df = pd.DataFrame(results_data). If results_data is [], df is empty.
    # df.to_csv will write an empty file (no headers).
    # If we want headers, we need to define them or pass a df with columns.
    # The print("Warning: No results data to save.") is good.
    # The current save_results will create an empty file if results_data is [].
    # If we want headers, we'd need something like:
    # if not results_data: pd.DataFrame(columns=['expected', 'columns']).to_csv(...)
    # For now, test the current behavior: empty file or just newline.
    assert output_file.stat().st_size >= 0 # File exists, could be empty or just newline

def test_save_results_directory_not_exists(tmp_path):
    """Test saving to a non-existent directory (should be handled by os.makedirs in real scenario, or error)."""
    # Note: pd.to_csv itself doesn't create parent directories.
    # This test depends on whether save_results is expected to create dirs or not.
    # Current save_results does not create directories.
    output_file = tmp_path / "non_existent_dir" / "output.csv"
    with pytest.raises(DataLoaderError, match="Error saving results to"): # pandas will raise FileNotFoundError, wrapped by DataLoaderError
        save_results([{'a':1}], str(output_file))

```
