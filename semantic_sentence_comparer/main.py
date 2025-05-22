import argparse
import logging
import os
import re
import dotenv

from src import data_loader as dl
from src import llm_scorers as sc

def sanitize_model_name(model_name: str) -> str:
    """Sanitizes model name for use as a CSV column header."""
    # Replace problematic characters (like '/') with underscores
    # Add '_score' suffix
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', model_name) + "_score"

def main():
    """
    Main function to orchestrate sentence similarity scoring.
    """
    # Load environment variables from .env file in the project root
    # This will be semantic_sentence_comparer/.env
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(dotenv_path):
        dotenv.load_dotenv(dotenv_path)
        logging.info(f"Loaded environment variables from {dotenv_path}")
    else:
        # Scorers might still work if env vars are set globally,
        # but it's good to note if .env is not found where expected.
        logging.warning(f".env file not found at {dotenv_path}. "
                        "API keys must be set in the environment globally if not using a .env file.")


    parser = argparse.ArgumentParser(description="Compare semantic similarity of sentences using LLMs.")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_csv", required=True, help="Path for the output CSV file.")
    parser.add_argument("--gold_column", required=True, help="Name of the column containing gold standard sentences.")
    parser.add_argument("--translated_column", required=True, help="Name of the column containing translated sentences.")
    parser.add_argument("--openai_model", default="gpt-3.5-turbo", help="OpenAI model name.")
    parser.add_argument("--gemini_model", default="gemini-pro", help="Gemini model name.")
    parser.add_argument("--hf_model", default="sentence-transformers/all-MiniLM-L6-v2", help="HuggingFace model name.")
    parser.add_argument("--scorers", nargs='+', default=['openai', 'gemini', 'hf'], 
                        choices=['openai', 'gemini', 'hf'], help="A list of scorers to use (e.g., --scorers openai hf).")
    parser.add_argument("--scorer_weights", nargs='+',
                        help="Optional: A list of scorer weights in the format 'scorer_name:weight' (e.g., openai:0.7 gemini:0.3). "
                             "Weights must be positive. If not provided, a simple average is used.")

    args = parser.parse_args()

    # Setup Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting semantic sentence comparison process.")
    logging.info(f"Input CSV: {args.input_csv}")
    logging.info(f"Output CSV: {args.output_csv}")
    logging.info(f"Gold Column: {args.gold_column}")
    logging.info(f"Translated Column: {args.translated_column}")
    logging.info(f"Requested Scorers: {args.scorers}")

    parsed_weights = {}
    if args.scorer_weights:
        logging.info(f"Processing scorer weights: {args.scorer_weights}")
        valid_scorer_names = {'openai', 'gemini', 'hf'}
        for item in args.scorer_weights:
            try:
                scorer_name, weight_str = item.split(':')
                if scorer_name not in valid_scorer_names:
                    logging.warning(f"Invalid scorer name '{scorer_name}' in --scorer_weights. It will be ignored. Must be one of {valid_scorer_names}.")
                    continue
                weight = float(weight_str)
                if weight <= 0:
                    logging.warning(f"Non-positive weight '{weight}' for scorer '{scorer_name}' found in --scorer_weights. This scorer might be excluded or handled by default averaging.")
                    # Storing it anyway for now, further validation might happen later or lead to fallback
                    parsed_weights[scorer_name] = weight 
                else:
                    parsed_weights[scorer_name] = weight
                    logging.info(f"Successfully parsed weight for {scorer_name}: {weight}")
            except ValueError:
                logging.warning(f"Incorrect format for scorer weight: '{item}'. Expected 'scorer_name:weight'. It will be ignored.")
            except Exception as e:
                logging.warning(f"An unexpected error occurred while parsing scorer weight '{item}': {e}. It will be ignored.")
        
        # Check if all requested scorers have weights if weights are provided
        if parsed_weights: # only if any weight was successfully parsed
            for scorer_name in args.scorers:
                if scorer_name not in parsed_weights:
                    logging.warning(f"Scorer '{scorer_name}' was requested but no weight was provided in --scorer_weights. "
                                    "This might lead to fallback to simple average or exclusion depending on later logic.")
    else:
        logging.info("No --scorer_weights provided. Will use simple average if multiple scorers are active.")

    # Initialize Scorers
    active_scorers = []
    scorer_map = {} # To store instances for later reference by name

    if 'openai' in args.scorers:
        try:
            openai_scorer = sc.OpenAIScorer(model_name=args.openai_model)
            active_scorers.append(openai_scorer)
            scorer_map['openai'] = openai_scorer
            logging.info(f"Initialized OpenAI scorer with model: {args.openai_model}")
        except sc.LLMScorerError as e:
            logging.error(f"Failed to initialize OpenAI scorer: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while initializing OpenAI scorer: {e}")

    if 'gemini' in args.scorers:
        try:
            gemini_scorer = sc.GeminiScorer(model_name=args.gemini_model)
            active_scorers.append(gemini_scorer)
            scorer_map['gemini'] = gemini_scorer
            logging.info(f"Initialized Gemini scorer with model: {args.gemini_model}")
        except sc.LLMScorerError as e:
            logging.error(f"Failed to initialize Gemini scorer: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while initializing Gemini scorer: {e}")

    if 'hf' in args.scorers:
        try:
            hf_scorer = sc.HuggingFaceScorer(model_name=args.hf_model)
            active_scorers.append(hf_scorer)
            scorer_map['hf'] = hf_scorer
            logging.info(f"Initialized HuggingFace scorer with model: {args.hf_model}")
        except sc.LLMScorerError as e:
            logging.error(f"Failed to initialize HuggingFace scorer: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while initializing HuggingFace scorer: {e}")

    if not active_scorers:
        logging.error("No scorers were successfully initialized. Exiting.")
        return

    # Load Data
    try:
        logging.info(f"Loading sentences from {args.input_csv}...")
        sentence_pairs = dl.load_sentences(args.input_csv, args.gold_column, args.translated_column)
        logging.info(f"Successfully loaded {len(sentence_pairs)} sentence pairs.")
    except dl.DataLoaderError as e:
        logging.error(f"Error loading data: {e}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}")
        return

    if not sentence_pairs:
        logging.warning("No sentence pairs loaded. Exiting.")
        # Attempt to save an empty results file with headers if output_csv is specified
        try:
            dl.save_results([], args.output_csv)
            logging.info(f"Empty results file saved to {args.output_csv}")
        except dl.DataLoaderError as e:
            logging.error(f"Error saving empty results file: {e}")
        return

    # Process Sentences
    all_results_data = []
    for i, pair in enumerate(sentence_pairs):
        logging.info(f"Processing sentence pair {i+1}/{len(sentence_pairs)}...")
        gold_sentence = pair['gold']
        translated_sentence = pair['translated']
        
        current_result = {
            'gold_sentence': gold_sentence,
            'translated_sentence': translated_sentence
        }
        pair_scores_data = [] # To store tuples of (score, weight_key)

        for scorer in active_scorers:
            # Determine the key for parsed_weights (e.g., 'openai', 'gemini', 'hf')
            scorer_type_key = None
            if isinstance(scorer, sc.OpenAIScorer): scorer_type_key = 'openai'
            elif isinstance(scorer, sc.GeminiScorer): scorer_type_key = 'gemini'
            elif isinstance(scorer, sc.HuggingFaceScorer): scorer_type_key = 'hf'
            
            scorer_name_log_key = scorer.model_name # For logging
            if isinstance(scorer, sc.OpenAIScorer): scorer_name_log_key = f"OpenAI_{scorer.model_name}"
            elif isinstance(scorer, sc.GeminiScorer): scorer_name_log_key = f"Gemini_{scorer.model_name}"
            elif isinstance(scorer, sc.HuggingFaceScorer): scorer_name_log_key = f"HF_{scorer.model_name}"

            sanitized_column_name = sanitize_model_name(scorer.model_name)
            
            try:
                logging.info(f"Scoring with {scorer_name_log_key}...")
                score = scorer.score(gold_sentence, translated_sentence)
                logging.info(f"Score from {scorer_name_log_key}: {score}")
                current_result[sanitized_column_name] = score
                pair_scores_data.append({'score': score, 'type_key': scorer_type_key, 'name_for_log': scorer_name_log_key})
            except Exception as e:
                logging.error(f"Error scoring with {scorer_name_log_key} for pair {i+1}: {e}. Assigning default score 1.0.")
                current_result[sanitized_column_name] = 1.0 # Default/penalty score
                pair_scores_data.append({'score': 1.0, 'type_key': scorer_type_key, 'name_for_log': scorer_name_log_key})

        # Calculate average score
        total_score = 0
        total_weight = 0
        use_weighted_average = False

        if parsed_weights: # Check if scorer_weights were provided and parsed
            temp_total_score = 0
            temp_total_weight = 0
            can_use_weighted_average_for_this_pair = True
            
            active_scorer_type_keys_in_pair = {psd['type_key'] for psd in pair_scores_data if psd['type_key']}

            for type_key in active_scorer_type_keys_in_pair:
                if type_key not in parsed_weights or parsed_weights[type_key] <= 0:
                    logging.warning(f"Weight for active scorer '{type_key}' is missing, zero, or negative in parsed_weights for pair {i+1}. "
                                    "Falling back to simple average for this pair.")
                    can_use_weighted_average_for_this_pair = False
                    break # Exit this inner loop, simple average will be used
            
            if can_use_weighted_average_for_this_pair:
                for psd in pair_scores_data:
                    weight = parsed_weights.get(psd['type_key'], 0) # Default to 0 if somehow not found after check
                    if weight > 0: # Ensure positive weight one last time
                       temp_total_score += psd['score'] * weight
                       temp_total_weight += weight
                    # If weight is not > 0 for an active scorer, can_use_weighted_average_for_this_pair should be false
                    # This path (weight <=0 here) should ideally not be hit if previous checks are robust
                
                if temp_total_weight > 0:
                    total_score = temp_total_score
                    total_weight = temp_total_weight
                    use_weighted_average = True
                    logging.info(f"Using weighted average for pair {i+1}.")
                else: # Should not happen if can_use_weighted_average_for_this_pair is true and weights were positive
                    logging.warning(f"Total weight is zero for weighted average calculation for pair {i+1}. Fallback to simple average.")
                    can_use_weighted_average_for_this_pair = False # Force simple average

        if not use_weighted_average: # Fallback to simple average
            logging.info(f"Using simple average for pair {i+1}.")
            simple_scores = [psd['score'] for psd in pair_scores_data]
            if simple_scores:
                total_score = sum(simple_scores)
                total_weight = len(simple_scores)
            else: # Should not happen if active_scorers is not empty
                total_score = 0
                total_weight = 1 # Avoid division by zero, avg_score will be 0

        avg_score = total_score / total_weight if total_weight > 0 else 0.0
        
        current_result['average_score'] = avg_score
        logging.info(f"Average score for pair {i+1}: {avg_score:.2f} ({'weighted' if use_weighted_average else 'simple'})")
        all_results_data.append(current_result)

    # Save Results
    try:
        logging.info(f"Saving results to {args.output_csv}...")
        dl.save_results(all_results_data, args.output_csv)
        logging.info(f"Processing complete. Results saved to {args.output_csv}")
    except dl.DataLoaderError as e:
        logging.error(f"Error saving results: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during results saving: {e}")

if __name__ == "__main__":
    # Configure basic logging first to catch early messages
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
