import pytest
import argparse
import logging
from unittest.mock import MagicMock, patch, call # call is useful for checking logger calls

# Assuming main.py is in the parent directory relative to the tests directory when tests are run from project root
# Or, if semantic_sentence_comparer is installed or PYTHONPATH is set, this would be simpler.
# For now, let's adjust path for direct execution if needed, but pytest handles this well.
from semantic_sentence_comparer import main as ssc_main # ssc_main to avoid conflict
from semantic_sentence_comparer.src import llm_scorers, data_loader

# Default sentence pair for tests
DEFAULT_SENTENCE_PAIRS = [
    {'gold': 'This is a gold sentence.', 'translated': 'This is a translated sentence.'}
]

# Default args that can be overridden by each test
def get_default_args():
    return argparse.Namespace(
        input_csv="dummy.csv",
        output_csv="dummy_out.csv",
        gold_column="gold",
        translated_column="translated",
        openai_model="gpt-3.5-turbo",
        gemini_model="gemini-pro",
        hf_model="sentence-transformers/all-MiniLM-L6-v2",
        scorers=['openai', 'hf'], # Default to two scorers for easier average calculation
        scorer_weights=None
    )

@pytest.fixture
def mock_data_loader(mocker):
    mocker.patch.object(data_loader, 'load_sentences', return_value=DEFAULT_SENTENCE_PAIRS)
    mock_save = mocker.patch.object(data_loader, 'save_results')
    return mock_save

@pytest.fixture
def mock_loggers(mocker):
    mock_info = mocker.patch.object(logging, 'info')
    mock_warning = mocker.patch.object(logging, 'warning')
    mock_error = mocker.patch.object(logging, 'error') # Good to mock error too
    return {'info': mock_info, 'warning': mock_warning, 'error': mock_error}

@pytest.fixture
def mock_scorers(mocker):
    # Mock OpenAIScorer
    mock_openai_scorer_instance = MagicMock(spec=llm_scorers.OpenAIScorer)
    mock_openai_scorer_instance.score.return_value = 4.0
    mock_openai_scorer_instance.model_name = "gpt-3.5-turbo" # Crucial for sanitize_model_name and logging
    mocker.patch.object(llm_scorers, 'OpenAIScorer', return_value=mock_openai_scorer_instance)

    # Mock GeminiScorer
    mock_gemini_scorer_instance = MagicMock(spec=llm_scorers.GeminiScorer)
    mock_gemini_scorer_instance.score.return_value = 3.0
    mock_gemini_scorer_instance.model_name = "gemini-pro"
    mocker.patch.object(llm_scorers, 'GeminiScorer', return_value=mock_gemini_scorer_instance)

    # Mock HuggingFaceScorer
    mock_hf_scorer_instance = MagicMock(spec=llm_scorers.HuggingFaceScorer)
    mock_hf_scorer_instance.score.return_value = 5.0 # Different score for HF
    mock_hf_scorer_instance.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    mocker.patch.object(llm_scorers, 'HuggingFaceScorer', return_value=mock_hf_scorer_instance)
    
    return {
        'openai': mock_openai_scorer_instance,
        'gemini': mock_gemini_scorer_instance,
        'hf': mock_hf_scorer_instance
    }

# Test Case 1: Simple Average (No Weights Provided)
def test_simple_average_no_weights(mocker, mock_data_loader, mock_scorers, mock_loggers):
    args = get_default_args()
    args.scorers = ['openai', 'hf'] # Explicitly using two scorers

    # Scores: openai=4.0, hf=5.0. Simple average = (4.0 + 5.0) / 2 = 4.5
    expected_avg_score = 4.5

    mocker.patch('argparse.ArgumentParser.parse_args', return_value=args)
    ssc_main.main()

    # Check that save_results was called
    assert mock_data_loader.called
    # Get the actual data passed to save_results
    saved_data = mock_data_loader.call_args[0][0]
    assert len(saved_data) == 1
    assert saved_data[0]['average_score'] == expected_avg_score
    mock_loggers['info'].assert_any_call(f"Using simple average for pair 1/1.")
    mock_loggers['info'].assert_any_call(f"Average score for pair 1/1: {expected_avg_score:.2f} (simple)")


# Test Case 2: Valid Weighted Average
def test_valid_weighted_average(mocker, mock_data_loader, mock_scorers, mock_loggers):
    args = get_default_args()
    args.scorers = ['openai', 'hf']
    args.scorer_weights = ['openai:0.6', 'hf:0.4']

    # Scores: openai=4.0, hf=5.0. Weighted: (4.0*0.6 + 5.0*0.4) / (0.6+0.4) = (2.4 + 2.0) / 1.0 = 4.4
    expected_avg_score = 4.4

    mocker.patch('argparse.ArgumentParser.parse_args', return_value=args)
    ssc_main.main()

    saved_data = mock_data_loader.call_args[0][0]
    assert len(saved_data) == 1
    assert saved_data[0]['average_score'] == pytest.approx(expected_avg_score)
    mock_loggers['info'].assert_any_call("Using weighted average for pair 1/1.")
    mock_loggers['info'].assert_any_call(f"Average score for pair 1/1: {expected_avg_score:.2f} (weighted)")

# Test Case 3: Weighted Average (Weights Don't Sum to 1)
def test_weighted_average_weights_not_sum_to_1(mocker, mock_data_loader, mock_scorers, mock_loggers):
    args = get_default_args()
    args.scorers = ['openai', 'hf']
    args.scorer_weights = ['openai:2', 'hf:3']

    # Scores: openai=4.0, hf=5.0. Weighted: (4.0*2 + 5.0*3) / (2+3) = (8.0 + 15.0) / 5.0 = 23.0 / 5.0 = 4.6
    expected_avg_score = 4.6

    mocker.patch('argparse.ArgumentParser.parse_args', return_value=args)
    ssc_main.main()

    saved_data = mock_data_loader.call_args[0][0]
    assert len(saved_data) == 1
    assert saved_data[0]['average_score'] == pytest.approx(expected_avg_score)
    mock_loggers['info'].assert_any_call("Using weighted average for pair 1/1.")
    mock_loggers['info'].assert_any_call(f"Average score for pair 1/1: {expected_avg_score:.2f} (weighted)")

# Test Case 4: Fallback to Simple Average (Missing Weight for an Active Scorer)
def test_fallback_simple_average_missing_weight(mocker, mock_data_loader, mock_scorers, mock_loggers):
    args = get_default_args()
    args.scorers = ['openai', 'hf'] # openai and hf are active
    args.scorer_weights = ['openai:0.5'] # Weight only for openai

    # Scores: openai=4.0, hf=5.0. Expected fallback to simple average = (4.0 + 5.0) / 2 = 4.5
    expected_avg_score = 4.5

    mocker.patch('argparse.ArgumentParser.parse_args', return_value=args)
    ssc_main.main()

    saved_data = mock_data_loader.call_args[0][0]
    assert len(saved_data) == 1
    assert saved_data[0]['average_score'] == expected_avg_score
    
    # Check for the warning about missing weight and fallback to simple average
    mock_loggers['warning'].assert_any_call(
        "Scorer 'hf' was requested but no weight was provided in --scorer_weights. "
        "This might lead to fallback to simple average or exclusion depending on later logic."
    )
    # This specific warning comes from the weight calculation loop
    mock_loggers['warning'].assert_any_call(
        "Weight for active scorer 'hf' is missing, zero, or negative in parsed_weights for pair 1/1. "
        "Falling back to simple average for this pair."
    )
    mock_loggers['info'].assert_any_call(f"Using simple average for pair 1/1.")
    mock_loggers['info'].assert_any_call(f"Average score for pair 1/1: {expected_avg_score:.2f} (simple)")


# Test Case 5: Fallback to Simple Average (Non-Positive Weight for an Active Scorer)
def test_fallback_simple_average_non_positive_weight(mocker, mock_data_loader, mock_scorers, mock_loggers):
    args = get_default_args()
    args.scorers = ['openai', 'hf']
    args.scorer_weights = ['openai:0.5', 'hf:0'] # HF weight is zero

    # Scores: openai=4.0, hf=5.0. Expected fallback to simple average = (4.0 + 5.0) / 2 = 4.5
    expected_avg_score = 4.5

    mocker.patch('argparse.ArgumentParser.parse_args', return_value=args)
    ssc_main.main()

    saved_data = mock_data_loader.call_args[0][0]
    assert len(saved_data) == 1
    assert saved_data[0]['average_score'] == expected_avg_score
    
    mock_loggers['warning'].assert_any_call(
        "Non-positive weight '0.0' for scorer 'hf' found in --scorer_weights. "
        "This scorer might be excluded or handled by default averaging."
    )
    mock_loggers['warning'].assert_any_call(
        "Weight for active scorer 'hf' is missing, zero, or negative in parsed_weights for pair 1/1. "
        "Falling back to simple average for this pair."
    )
    mock_loggers['info'].assert_any_call(f"Using simple average for pair 1/1.")
    mock_loggers['info'].assert_any_call(f"Average score for pair 1/1: {expected_avg_score:.2f} (simple)")

# Test Case 6: Handling of Invalid Weight Format in Argument
def test_invalid_weight_format_logged_fallback_simple(mocker, mock_data_loader, mock_scorers, mock_loggers):
    args = get_default_args()
    args.scorers = ['openai', 'hf']
    args.scorer_weights = ['openai0.5', 'hf:0.8'] # First weight is malformed

    # Scores: openai=4.0, hf=5.0. 
    # openai0.5 is ignored. hf:0.8 is valid.
    # Since not all active scorers (openai, hf) have valid weights, it should fall back to simple.
    expected_avg_score = 4.5 

    mocker.patch('argparse.ArgumentParser.parse_args', return_value=args)
    ssc_main.main()

    saved_data = mock_data_loader.call_args[0][0]
    assert len(saved_data) == 1
    assert saved_data[0]['average_score'] == expected_avg_score

    mock_loggers['warning'].assert_any_call(
        "Incorrect format for scorer weight: 'openai0.5'. Expected 'scorer_name:weight'. It will be ignored."
    )
    # This warning is because 'openai' is active but its weight 'openai0.5' was invalid and ignored.
    # So 'openai' does not appear in parsed_weights.
    mock_loggers['warning'].assert_any_call(
        "Scorer 'openai' was requested but no weight was provided in --scorer_weights. "
        "This might lead to fallback to simple average or exclusion depending on later logic."
    )
    # This warning is from the calculation loop because openai's weight is missing from parsed_weights.
    mock_loggers['warning'].assert_any_call(
        "Weight for active scorer 'openai' is missing, zero, or negative in parsed_weights for pair 1/1. "
        "Falling back to simple average for this pair."
    )
    mock_loggers['info'].assert_any_call(f"Using simple average for pair 1/1.")
    mock_loggers['info'].assert_any_call(f"Average score for pair 1/1: {expected_avg_score:.2f} (simple)")

# Test Case 7: Only some scorers have weights (more than 2 scorers active)
def test_fallback_simple_avg_some_scorers_have_weights(mocker, mock_data_loader, mock_scorers, mock_loggers):
    args = get_default_args()
    args.scorers = ['openai', 'gemini', 'hf'] # Three active scorers
    args.scorer_weights = ['openai:0.7', 'gemini:0.3'] # Weights only for two

    # Scores: openai=4.0, gemini=3.0, hf=5.0
    # Expected fallback to simple average = (4.0 + 3.0 + 5.0) / 3 = 12.0 / 3 = 4.0
    expected_avg_score = 4.0

    mocker.patch('argparse.ArgumentParser.parse_args', return_value=args)
    ssc_main.main()

    saved_data = mock_data_loader.call_args[0][0]
    assert len(saved_data) == 1
    assert saved_data[0]['average_score'] == pytest.approx(expected_avg_score)

    mock_loggers['warning'].assert_any_call(
        "Scorer 'hf' was requested but no weight was provided in --scorer_weights. "
        "This might lead to fallback to simple average or exclusion depending on later logic."
    )
    mock_loggers['warning'].assert_any_call(
        "Weight for active scorer 'hf' is missing, zero, or negative in parsed_weights for pair 1/1. "
        "Falling back to simple average for this pair."
    )
    mock_loggers['info'].assert_any_call(f"Using simple average for pair 1/1.")
    mock_loggers['info'].assert_any_call(f"Average score for pair 1/1: {expected_avg_score:.2f} (simple)")

# Test for invalid scorer name in weights
def test_invalid_scorer_name_in_weights(mocker, mock_data_loader, mock_scorers, mock_loggers):
    args = get_default_args()
    args.scorers = ['openai', 'hf']
    args.scorer_weights = ['openai:0.7', 'unknown_scorer:0.3']

    # Scores: openai=4.0, hf=5.0. 'unknown_scorer' is ignored.
    # Fallback to simple because 'hf' is active but has no valid weight.
    expected_avg_score = 4.5

    mocker.patch('argparse.ArgumentParser.parse_args', return_value=args)
    ssc_main.main()

    saved_data = mock_data_loader.call_args[0][0]
    assert len(saved_data) == 1
    assert saved_data[0]['average_score'] == pytest.approx(expected_avg_score)

    mock_loggers['warning'].assert_any_call(
        "Invalid scorer name 'unknown_scorer' in --scorer_weights. It will be ignored. Must be one of {'openai', 'gemini', 'hf'}."
    )
    mock_loggers['warning'].assert_any_call(
        "Scorer 'hf' was requested but no weight was provided in --scorer_weights. "
        "This might lead to fallback to simple average or exclusion depending on later logic."
    )
    mock_loggers['warning'].assert_any_call(
        "Weight for active scorer 'hf' is missing, zero, or negative in parsed_weights for pair 1/1. "
        "Falling back to simple average for this pair."
    )
    mock_loggers['info'].assert_any_call(f"Using simple average for pair 1/1.")
    mock_loggers['info'].assert_any_call(f"Average score for pair 1/1: {expected_avg_score:.2f} (simple)")

# Test case where no scorers are successfully initialized
def test_no_active_scorers(mocker, mock_data_loader, mock_loggers):
    args = get_default_args()
    args.scorers = ['openai'] # Request one scorer

    # Mock OpenAIScorer __init__ to raise an error
    mocker.patch.object(llm_scorers, 'OpenAIScorer', side_effect=llm_scorers.LLMScorerError("Initialization failed"))
    
    mocker.patch('argparse.ArgumentParser.parse_args', return_value=args)
    ssc_main.main()

    mock_loggers['error'].assert_any_call("Failed to initialize OpenAI scorer: Initialization failed")
    mock_loggers['error'].assert_any_call("No scorers were successfully initialized. Exiting.")
    # Ensure save_results is not called with actual data, or if it is, it's with an empty list
    # In current main.py, it returns after logging "No scorers were successfully initialized."
    # So save_results should not be called for processing, but an empty results file might be saved
    # For this test, let's verify it doesn't proceed to process sentences.
    # We can check that load_sentences was not called after the point of scorer initialization failure.
    # However, the current structure of main.py initializes scorers before loading data.
    # If no scorers, it exits. So mock_data_loader (for save_results) should not have been called with actual results.
    
    # Check if save_results was called (it might be called with an empty list if output_csv is specified,
    # but the current main.py exits before that if no scorers).
    # Based on current main.py, if no active_scorers, it returns.
    # Let's ensure it didn't try to save non-empty data.
    for call_item in mock_data_loader.call_args_list:
        saved_data = call_item[0][0]
        assert len(saved_data) == 0 # Should be empty if called
    
    # More precise: check that specific info logs about processing pairs are NOT present
    processing_log_found = False
    for call_item in mock_loggers['info'].call_args_list:
        if "Processing sentence pair" in call_item[0][0]:
            processing_log_found = True
            break
    assert not processing_log_found, "Should not process sentence pairs if no scorers are active."
