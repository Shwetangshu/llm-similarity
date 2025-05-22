import pytest
import os
from unittest.mock import MagicMock, patch

# Ensure .env is loaded for tests if it exists, or environment variables are set
# This path assumes tests are run from the project root (semantic_sentence_comparer/)
# or that the path to .env is correctly resolved.
from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env') # Adjust if your .env is elsewhere or tests run differently
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else: # Fallback for CI or environments where .env is not in the parent dir of tests/
    dotenv_path_alt = os.path.join(os.path.dirname(__file__), '..', '..', '.env') # semantic_sentence_comparer/.env
    if os.path.exists(dotenv_path_alt):
        load_dotenv(dotenv_path_alt)


from src.llm_scorers import (
    OpenAIScorer, 
    GeminiScorer, 
    HuggingFaceScorer, 
    LLMScorerError,
    LLMScorer # For _extract_score, need an instance
)

# Need an instance of a class that has _extract_score
# We can use any concrete class that inherits from LLMScorer for this.
# Let's use OpenAIScorer as a dummy for testing _extract_score.
# This requires OPENAI_API_KEY to be set or mocked during this dummy instantiation.
# To avoid issues, we can temporarily patch os.getenv for this specific setup.
with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key_for_extractor"}):
    try:
        dummy_scorer_for_extract_test = OpenAIScorer(model_name="dummy_for_test")
    except LLMScorerError: # If API key is really required for the client init itself
        dummy_scorer_for_extract_test = LLMScorer(model_name="dummy_for_test") # ABC instance
        # If LLMScorer can't be instantiated, create a simple dummy class
        class DummyExtractor(LLMScorer):
            def score(self,s1,s2): return 0.0
        dummy_scorer_for_extract_test = DummyExtractor("dummy")


@pytest.mark.parametrize("response_text, expected_score", [
    ("Score: 4.5", 4.5),
    ("The final score is 3.2.", 3.2),
    ("The score is 3.", 3.0),
    ("Rating: 5", 5.0),
    ("5", 5.0),
    ("2.0", 2.0),
    ("Rating: 1 out of 5", 1.0),
    ("The model rates this as a 4 out of 5.", 4.0),
    ("This is a 1.0 on the scale.", 1.0),
    ("The similarity is high, around 4.8.", 4.8),
    ("Score: 7", 7.0), # Test out-of-range extraction (clamping is done in scorer methods)
    ("Score: 0.5", 0.5), # Test out-of-range extraction
    ("No score here", None),
    ("The score is five.", None), # Not numerical
    ("The score is 3 and a half", 3.0), # Should pick 3
    ("Response with 2 numbers 4.0 and 5.0", 4.0) # Picks first valid
])
def test_extract_score(response_text, expected_score):
    """Test the _extract_score method with various inputs."""
    # Note: _extract_score is called by an instance of a class inheriting LLMScorer
    # We use a dummy instance here.
    assert dummy_scorer_for_extract_test._extract_score(response_text) == expected_score

# --- OpenAIScorer Tests ---
def test_openai_scorer_init_success(mocker):
    """Test OpenAIScorer instantiation success with mocked API key."""
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"})
    scorer = OpenAIScorer(model_name="test-gpt")
    assert scorer.model_name == "test-gpt"
    assert scorer.client is not None

def test_openai_scorer_init_no_api_key(mocker):
    """Test OpenAIScorer instantiation fails if API key is missing."""
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": ""}) # Empty key
    with pytest.raises(LLMScorerError, match="OPENAI_API_KEY not found"):
        OpenAIScorer()
    
    mocker.patch.dict(os.environ)
    if "OPENAI_API_KEY" in os.environ: # Ensure it's fully removed
        del os.environ["OPENAI_API_KEY"]
    with pytest.raises(LLMScorerError, match="OPENAI_API_KEY not found"):
        OpenAIScorer()

@patch('openai.OpenAI') # Mocks the OpenAI client constructor
def test_openai_scorer_score_success(MockOpenAIClient, mocker):
    """Test OpenAIScorer score method successfully extracts score from mocked API response."""
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"})
    
    mock_completion_response = MagicMock()
    mock_completion_response.choices[0].message.content = "The similarity score is 4.2."
    
    mock_openai_instance = MockOpenAIClient.return_value # Get the instance of the mocked client
    mock_openai_instance.chat.completions.create.return_value = mock_completion_response

    scorer = OpenAIScorer(model_name="gpt-3.5-turbo")
    score = scorer.score("Sentence 1", "Sentence 2")
    
    assert score == 4.2
    mock_openai_instance.chat.completions.create.assert_called_once()

@patch('openai.OpenAI')
def test_openai_scorer_score_api_error(MockOpenAIClient, mocker):
    """Test OpenAIScorer score method handles API errors gracefully."""
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"})
    
    mock_openai_instance = MockOpenAIClient.return_value
    mock_openai_instance.chat.completions.create.side_effect = Exception("OpenAI API Error")

    scorer = OpenAIScorer(model_name="gpt-3.5-turbo")
    score = scorer.score("Sentence 1", "Sentence 2")
    
    assert score == 1.0 # Default score on error

@patch('openai.OpenAI')
def test_openai_scorer_score_no_valid_score_in_response(MockOpenAIClient, mocker):
    """Test OpenAIScorer score method handles responses with no extractable score."""
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"})
    
    mock_completion_response = MagicMock()
    mock_completion_response.choices[0].message.content = "This is an unrelated response."
    
    mock_openai_instance = MockOpenAIClient.return_value
    mock_openai_instance.chat.completions.create.return_value = mock_completion_response

    scorer = OpenAIScorer(model_name="gpt-3.5-turbo")
    score = scorer.score("Sentence 1", "Sentence 2")
    
    assert score == 1.0 # Default score when extraction fails

# --- GeminiScorer Tests ---
def test_gemini_scorer_init_success(mocker):
    """Test GeminiScorer instantiation success with mocked API key."""
    mocker.patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"})
    mocker.patch('google.generativeai.configure') # Mock configure call
    mocker.patch('google.generativeai.GenerativeModel') # Mock model initialization
    scorer = GeminiScorer(model_name="test-gemini")
    assert scorer.model_name == "test-gemini"
    assert scorer.model is not None
    google.generativeai.configure.assert_called_once_with(api_key="fake_key")

def test_gemini_scorer_init_no_api_key(mocker):
    """Test GeminiScorer instantiation fails if API key is missing."""
    mocker.patch.dict(os.environ, {"GOOGLE_API_KEY": ""})
    with pytest.raises(LLMScorerError, match="GOOGLE_API_KEY not found"):
        GeminiScorer()

    mocker.patch.dict(os.environ)
    if "GOOGLE_API_KEY" in os.environ:
        del os.environ["GOOGLE_API_KEY"]
    with pytest.raises(LLMScorerError, match="GOOGLE_API_KEY not found"):
        GeminiScorer()


@patch('google.generativeai.GenerativeModel')
def test_gemini_scorer_score_success(MockGenerativeModel, mocker):
    """Test GeminiScorer score method successfully extracts score from mocked API response."""
    mocker.patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"})
    mocker.patch('google.generativeai.configure')

    mock_gemini_response = MagicMock()
    mock_gemini_response.text = "Score: 3.8"
    
    mock_model_instance = MockGenerativeModel.return_value
    mock_model_instance.generate_content.return_value = mock_gemini_response

    scorer = GeminiScorer(model_name="gemini-pro")
    score = scorer.score("Sentence A", "Sentence B")
    
    assert score == 3.8
    mock_model_instance.generate_content.assert_called_once()

@patch('google.generativeai.GenerativeModel')
def test_gemini_scorer_score_api_error(MockGenerativeModel, mocker):
    """Test GeminiScorer score method handles API errors gracefully."""
    mocker.patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"})
    mocker.patch('google.generativeai.configure')

    mock_model_instance = MockGenerativeModel.return_value
    mock_model_instance.generate_content.side_effect = Exception("Gemini API Error")

    scorer = GeminiScorer(model_name="gemini-pro")
    score = scorer.score("Sentence A", "Sentence B")
    
    assert score == 1.0 # Default score on error

# --- HuggingFaceScorer Tests ---
@patch('transformers.AutoTokenizer.from_pretrained')
@patch('transformers.AutoModel.from_pretrained')
def test_hf_scorer_init_success(MockAutoModel, MockAutoTokenizer, mocker):
    """Test HuggingFaceScorer instantiation success with mocked model/tokenizer loading."""
    mock_tokenizer_instance = MagicMock()
    MockAutoTokenizer.return_value = mock_tokenizer_instance
    
    mock_model_instance = MagicMock()
    mock_model_instance.to.return_value = mock_model_instance # for .to(device)
    mock_model_instance.eval.return_value = None
    MockAutoModel.return_value = mock_model_instance
    
    scorer = HuggingFaceScorer(model_name="test-hf-model")
    assert scorer.model_name == "test-hf-model"
    assert scorer.tokenizer is not None
    assert scorer.model is not None
    MockAutoTokenizer.assert_called_once_with("test-hf-model")
    MockAutoModel.assert_called_once_with("test-hf-model")
    mock_model_instance.to.assert_called_once() # Check if .to(device) was called
    mock_model_instance.eval.assert_called_once()


@patch('transformers.AutoTokenizer.from_pretrained')
@patch('transformers.AutoModel.from_pretrained')
def test_hf_scorer_init_model_load_error(MockAutoModel, MockAutoTokenizer, mocker):
    """Test HuggingFaceScorer handles errors during model loading."""
    MockAutoModel.from_pretrained.side_effect = Exception("Model loading failed")
    
    with pytest.raises(LLMScorerError, match="Error loading Hugging Face model"):
        HuggingFaceScorer(model_name="failing-hf-model")

@patch('transformers.AutoTokenizer.from_pretrained')
@patch('transformers.AutoModel.from_pretrained')
@patch('torch.nn.functional.cosine_similarity')
@patch('torch.Tensor.unsqueeze') # To control the tensor returned by unsqueeze
@patch('torch.Tensor.item') # To control the float value from .item()
def test_hf_scorer_score_scaling(mock_item, mock_unsqueeze, mock_cosine_similarity, MockAutoModel, MockAutoTokenizer, mocker):
    """Test HuggingFaceScorer score method correctly scales cosine similarity."""
    # --- Setup Mocks ---
    mock_tokenizer_instance = MagicMock()
    # Simulate tokenizer output: needs 'input_ids' and 'attention_mask'
    mock_encoded_input = {
        'input_ids': MagicMock(), # A mock tensor
        'attention_mask': MagicMock() # A mock tensor
    }
    mock_tokenizer_instance.return_value = mock_encoded_input
    mock_tokenizer_instance.to.return_value = mock_tokenizer_instance # for .to(device)
    MockAutoTokenizer.return_value = mock_tokenizer_instance
    
    mock_model_instance = MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    # Simulate model output: a tuple where the first element is token embeddings
    mock_model_output = (MagicMock(),) # Mock tensor for token_embeddings
    mock_model_instance.return_value = mock_model_output # model(**encoded_input)
    MockAutoModel.return_value = mock_model_instance

    # Mock the _mean_pooling to return dummy embeddings
    # Need to patch the method on the class instance for the test
    mocker.patch.object(HuggingFaceScorer, '_mean_pooling', return_value=MagicMock()) # Returns a mock tensor

    # Mock unsqueeze to return a distinct mock for each call if needed, or just the same one
    mock_embedding_tensor = MagicMock()
    mock_unsqueeze.return_value = mock_embedding_tensor
    
    scorer = HuggingFaceScorer(model_name="test-hf-model") # Instantiation uses the mocks above

    test_cases = [
        (1.0, 5.0),  # Cosine similarity 1.0  -> Expected score 5.0
        (0.5, 4.0),  # Cosine similarity 0.5  -> Expected score 4.0
        (0.0, 3.0),  # Cosine similarity 0.0  -> Expected score 3.0
        (-0.5, 2.0), # Cosine similarity -0.5 -> Expected score 2.0
        (-1.0, 1.0)  # Cosine similarity -1.0 -> Expected score 1.0
    ]

    for cosine_sim, expected_scaled_score in test_cases:
        # Configure cosine_similarity mock for this iteration
        # It should return a tensor whose .item() method gives cosine_sim
        mock_similarity_tensor = MagicMock()
        mock_item.return_value = cosine_sim # mock_similarity_tensor.item() will be cosine_sim
        mock_cosine_similarity.return_value = mock_similarity_tensor

        # Call score method
        actual_score = scorer.score("Sentence X", "Sentence Y")
        
        # Assertions
        assert actual_score == expected_scaled_score
        # Check if tokenizer and model were called (they are part of setup for each score call)
        mock_tokenizer_instance.assert_called_with(["Sentence X", "Sentence Y"], padding=True, truncation=True, return_tensors='pt')
        # Check if cosine similarity was called with the (mocked) embeddings
        # This requires knowing what _mean_pooling returns. Since it's also mocked, this is a bit indirect.
        # We can check it was called.
        mock_cosine_similarity.assert_called_with(mock_embedding_tensor, mock_embedding_tensor)


@patch('transformers.AutoTokenizer.from_pretrained')
@patch('transformers.AutoModel.from_pretrained')
def test_hf_scorer_score_inference_error(MockAutoModel, MockAutoTokenizer, mocker):
    """Test HuggingFaceScorer handles errors during model inference gracefully."""
    mock_tokenizer_instance = MagicMock()
    MockAutoTokenizer.return_value = mock_tokenizer_instance
    
    mock_model_instance = MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.side_effect = Exception("HF Model Inference Error") # model(**encoded_input) raises error
    MockAutoModel.return_value = mock_model_instance

    scorer = HuggingFaceScorer(model_name="test-hf-model")
    score = scorer.score("Sentence X", "Sentence Y")
    
    assert score == 1.0 # Default score on error

```
