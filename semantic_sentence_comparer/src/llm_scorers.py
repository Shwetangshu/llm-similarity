import abc
import os
import re
import openai
import google.generativeai
import transformers
import torch
import dotenv
from typing import Optional, Tuple

# Load environment variables from .env in the project root
# Assumes this file is in semantic_sentence_comparer/src/
dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

class LLMScorerError(Exception):
    """Custom exception for LLMScorer related errors."""
    pass

class LLMScorer(abc.ABC):
    """
    Abstract base class for LLM-based sentence similarity scorers.
    """
    def __init__(self, model_name: str):
        """
        Initializes the scorer with a model name.

        Args:
            model_name: The name of the model to be used.
        """
        self.model_name = model_name

    @abc.abstractmethod
    def score(self, sentence1: str, sentence2: str) -> float:
        """
        Scores the semantic similarity between two sentences.

        Args:
            sentence1: The first sentence.
            sentence2: The second sentence.

        Returns:
            A similarity score, ideally between 1.0 and 5.0.
        """
        pass

    def _extract_score(self, text_response: str, min_score: float = 1.0, max_score: float = 5.0) -> Optional[float]:
        """
        Extracts a numerical score from a text response using regex.

        Args:
            text_response: The text response from the LLM.
            min_score: The minimum acceptable score.
            max_score: The maximum acceptable score.

        Returns:
            The extracted score as a float if found and valid, otherwise None.
        """
        if not text_response:
            print("Warning: Received empty text response for score extraction.")
            return None

        # Regex to find numbers (integers or decimals, including those like "Score: X" or just "X")
        # It tries to find numbers that might be preceded by "Score:", "Rating:", etc., or standalone numbers.
        patterns = [
            r"score:\s*([0-9]*\.?[0-9]+)",  # Matches "score: X.X" or "score: X"
            r"rating:\s*([0-9]*\.?[0-9]+)", # Matches "rating: X.X" or "rating: X"
            r"([0-9]*\.?[0-9]+)"            # Matches any number X.X or X
        ]
        
        found_score = None
        for pattern in patterns:
            match = re.search(pattern, text_response, re.IGNORECASE)
            if match:
                try:
                    # If the pattern has a capturing group for the number, use it.
                    # Otherwise, the whole match is the number.
                    score_str = match.group(1) if len(match.groups()) > 0 else match.group(0)
                    score = float(score_str)
                    found_score = score
                    break # Found a score, stop searching
                except ValueError:
                    print(f"Warning: Could not convert found text '{score_str}' to float.")
                    continue # Try next pattern
        
        if found_score is not None:
            if min_score <= found_score <= max_score:
                return found_score
            else:
                print(f"Warning: Extracted score {found_score} is outside the valid range [{min_score}, {max_score}]. Returning it anyway.")
                return found_score # Return score even if out of range, caller can decide
        else:
            print(f"Warning: No numerical score found in response: '{text_response}'")
            return None


class OpenAIScorer(LLMScorer):
    """
    Scores sentence similarity using OpenAI's models.
    """
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initializes the OpenAIScorer.

        Args:
            model_name: The OpenAI model name (e.g., "gpt-3.5-turbo", "gpt-4").
        """
        super().__init__(model_name)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMScorerError("OPENAI_API_KEY not found in environment variables.")
        self.client = openai.OpenAI(api_key=api_key)

    def score(self, sentence1: str, sentence2: str) -> float:
        """
        Scores the semantic similarity between two sentences using an OpenAI model.

        Args:
            sentence1: The gold standard sentence.
            sentence2: The translated sentence.

        Returns:
            A similarity score between 1.0 and 5.0. Returns 1.0 on failure.
        """
        system_prompt = (
            "You are an expert in semantic text analysis. Your task is to evaluate the "
            "semantic similarity between two sentences: a 'gold standard' sentence and a "
            "'translated' sentence. Provide a score from 1 to 5, where 1 means "
            "completely different meaning and 5 means identical meaning. "
            "Output only the numerical score."
        )
        user_prompt = (
            f'Gold standard: "{sentence1}"\n'
            f'Translated: "{sentence2}"\n'
            f'Score (1-5):'
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2, # Lower temperature for more deterministic scoring
                max_tokens=10 
            )
            text_response = response.choices[0].message.content.strip()
            extracted_score = self._extract_score(text_response)
            if extracted_score is not None:
                return max(1.0, min(5.0, extracted_score)) # Clamp to [1,5]
            else:
                print(f"Warning: Failed to extract score from OpenAI response: '{text_response}'. Returning default score 1.0.")
                return 1.0
        except Exception as e:
            print(f"Error during OpenAI API call: {e}. Returning default score 1.0.")
            return 1.0


class GeminiScorer(LLMScorer):
    """
    Scores sentence similarity using Google's Gemini models.
    """
    def __init__(self, model_name: str = "gemini-pro"):
        """
        Initializes the GeminiScorer.

        Args:
            model_name: The Gemini model name (e.g., "gemini-pro").
        """
        super().__init__(model_name)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LLMScorerError("GOOGLE_API_KEY not found in environment variables.")
        google.generativeai.configure(api_key=api_key)
        self.model = google.generativeai.GenerativeModel(model_name)

    def score(self, sentence1: str, sentence2: str) -> float:
        """
        Scores the semantic similarity between two sentences using a Gemini model.

        Args:
            sentence1: The gold standard sentence.
            sentence2: The translated sentence.

        Returns:
            A similarity score between 1.0 and 5.0. Returns 1.0 on failure.
        """
        prompt = (
            "Evaluate the semantic similarity between the following two sentences. "
            "Provide a score from 1 to 5, where 1 means completely different meaning "
            "and 5 means identical meaning. Output only the numerical score.\n\n"
            f'Sentence 1 (Gold Standard): "{sentence1}"\n'
            f'Sentence 2 (Translated): "{sentence2}"\n\n'
            "Score (1-5):"
        )

        try:
            # For gemini, generation_config can be helpful
            generation_config = google.generativeai.types.GenerationConfig(
                candidate_count=1,
                temperature=0.2, # Lower temperature for more deterministic scoring
                max_output_tokens=10 
            )
            response = self.model.generate_content(prompt, generation_config=generation_config)
            text_response = response.text.strip()
            extracted_score = self._extract_score(text_response)
            if extracted_score is not None:
                return max(1.0, min(5.0, extracted_score)) # Clamp to [1,5]
            else:
                print(f"Warning: Failed to extract score from Gemini response: '{text_response}'. Returning default score 1.0.")
                return 1.0
        except Exception as e:
            print(f"Error during Gemini API call: {e}. Returning default score 1.0.")
            return 1.0


class HuggingFaceScorer(LLMScorer):
    """
    Scores sentence similarity using Hugging Face sentence transformer models.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the HuggingFaceScorer.

        Args:
            model_name: The Hugging Face model name (e.g., "sentence-transformers/all-MiniLM-L6-v2").
        """
        super().__init__(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"HuggingFaceScorer: Using device: {self.device}")
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            self.model = transformers.AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval() # Set model to evaluation mode
        except Exception as e:
            raise LLMScorerError(f"Error loading Hugging Face model '{model_name}': {e}")

    def _mean_pooling(self, model_output: Tuple[torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs mean pooling on token embeddings.

        Args:
            model_output: Output from the transformer model.
            attention_mask: Attention mask for the input tokens.

        Returns:
            Sentence embedding tensor.
        """
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def score(self, sentence1: str, sentence2: str) -> float:
        """
        Scores the semantic similarity using sentence embeddings and cosine similarity.
        The similarity is scaled from [-1, 1] to [1, 5].

        Args:
            sentence1: The first sentence.
            sentence2: The second sentence.

        Returns:
            A scaled similarity score between 1.0 and 5.0.
        """
        sentences = [sentence1, sentence2]
        try:
            encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            embedding1 = sentence_embeddings[0].unsqueeze(0) # Ensure 2D for cosine_similarity
            embedding2 = sentence_embeddings[1].unsqueeze(0) # Ensure 2D for cosine_similarity

            similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2).item() # .item() to get Python float

            # Scale similarity from [-1, 1] to [1, 5]
            # -1 maps to 1: 1 + (-1 + 1) * 2 = 1 + 0 * 2 = 1
            #  0 maps to 3: 1 + (0 + 1) * 2  = 1 + 1 * 2 = 3
            #  1 maps to 5: 1 + (1 + 1) * 2  = 1 + 2 * 2 = 5
            scaled_score = 1 + (similarity + 1) * 2
            return max(1.0, min(5.0, scaled_score)) # Ensure it's strictly within [1,5]

        except Exception as e:
            print(f"Error during Hugging Face model inference or scoring: {e}. Returning default score 1.0.")
            return 1.0

if __name__ == '__main__':
    # Example Usage (requires .env file with API keys in project root)
    # Ensure you have semantic_sentence_comparer/.env
    
    print("Attempting to load .env from:", os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
    # Test OpenAI
    try:
        openai_scorer = OpenAIScorer()
        score_oai = openai_scorer.score("The cat sat on the mat.", "A feline was resting on the rug.")
        print(f"OpenAI Score: {score_oai}")
        score_oai_diff = openai_scorer.score("The sky is blue.", "Apples are red.")
        print(f"OpenAI Score (different): {score_oai_diff}")
        score_oai_typo = openai_scorer.score("This is a test sentence.", "This is a tset sentenc.") # Test typo
        print(f"OpenAI Score (typo): {score_oai_typo}")

    except LLMScorerError as e:
        print(f"OpenAI Scorer Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with OpenAIScorer: {e}")

    # Test Gemini
    try:
        gemini_scorer = GeminiScorer()
        score_gem = gemini_scorer.score("The weather is sunny today.", "It's a bright day with lots of sunshine.")
        print(f"Gemini Score: {score_gem}")
        score_gem_diff = gemini_scorer.score("He loves to play guitar.", "She enjoys reading books.")
        print(f"Gemini Score (different): {score_gem_diff}")
    except LLMScorerError as e:
        print(f"Gemini Scorer Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with GeminiScorer: {e}")

    # Test HuggingFace
    try:
        hf_scorer = HuggingFaceScorer() # Uses default "sentence-transformers/all-MiniLM-L6-v2"
        score_hf = hf_scorer.score("The quick brown fox jumps over the lazy dog.", "A fast, dark-colored fox leaps above a sleepy canine.")
        print(f"HuggingFace Score: {score_hf}")
        score_hf_identical = hf_scorer.score("This is a test.", "This is a test.")
        print(f"HuggingFace Score (identical): {score_hf_identical}")
        score_hf_diff = hf_scorer.score("The quick brown fox.", "The lazy dog.")
        print(f"HuggingFace Score (different): {score_hf_diff}")

    except LLMScorerError as e:
        print(f"HuggingFace Scorer Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with HuggingFaceScorer: {e}")

    # Test _extract_score
    print("\nTesting _extract_score method:")
    scorer_instance_for_test = OpenAIScorer() # Need an instance to call _extract_score
    
    test_responses = [
        ("Score: 4.5", 4.5),
        ("The final score is 3.2.", 3.2),
        ("Rating: 5", 5.0),
        ("2.5", 2.5),
        ("The model rates this as a 4 out of 5.", 4.0), # Should pick 4
        ("This is a 1.0 on the scale.", 1.0),
        ("The similarity is high, around 4.8.", 4.8),
        ("No score here", None),
        ("Score: 6.0", 6.0), # Out of typical range, but should be extracted
        ("Score: 0.5", 0.5), # Out of typical range
        ("The score is five.", None), # Not numerical
        ("The score is 3 and a half", 3.0), # Should pick 3
        ("Response with 2 numbers 4.0 and 5.0", 4.0) # Picks first
    ]

    for response_text, expected in test_responses:
        extracted = scorer_instance_for_test._extract_score(response_text)
        print(f"Input: '{response_text}', Extracted: {extracted}, Expected: {expected} -> {'Pass' if extracted == expected else 'Fail'}")

    # Test with a custom range
    extracted_custom_range = scorer_instance_for_test._extract_score("Score: 75", min_score=0, max_score=100)
    print(f"Input: 'Score: 75', Custom Range [0,100], Extracted: {extracted_custom_range} -> {'Pass' if extracted_custom_range == 75.0 else 'Fail'}")

```
