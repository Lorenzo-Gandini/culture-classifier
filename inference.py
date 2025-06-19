import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import editdistance

from ocr_dataset import retrieve_datasets


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_model_and_tokenizer(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer from path."""
    logging.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logging.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    return model, tokenizer


def load_test_data(dataset_path: str, max_samples: int = None) -> List[Tuple[str, str]]:
    """Load and prepare test dataset."""
    logging.info(f"Loading dataset from {dataset_path}")
    _, test_data = retrieve_datasets(dataset_path, test_size=0.5)
    test_samples = list(test_data)
    
    if max_samples:
        test_samples = test_samples[:max_samples]
        logging.info(f"Limited to {max_samples} samples")
    
    logging.info(f"Loaded {len(test_samples)} test samples")
    return test_samples


def calculate_metrics(prediction: str, target: str) -> Dict[str, float]:
    """Calculate CER and WER metrics."""
    cer = editdistance.eval(prediction, target) / max(len(target), 1)
    
    pred_words = prediction.split()
    target_words = target.split()
    wer = editdistance.eval(pred_words, target_words) / max(len(target_words), 1)
    
    return {"cer": cer, "wer": wer}


def generate_prediction(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_text: str,
    max_new_tokens: int = 100
) -> str:
    
    few_shot = (
    "These are few examples that u might use\n"
    "<|user|> Thls ls a test of the emergeney broadcast systern. Thls ls only a test.\n"
    "<|assistant|> This is a test of the emergency broadcast system. This is only a test.\n\n"

    "<|user|>: The quick brown fox jumps ovor the laay dog.\n"
    "<|assistant|>: The quick brown fox jumps over the lazy dog.\n\n"

    "<|user|>: lt was the best of tirnes, lt was the worst of tirnes.\n"
    "<|assistant|>: It was the best of times, it was the worst of times.\n\n"
    )


    prompt = few_shot + f"<|user|>: {input_text}\n<|assistant|>:"

    """Generate prediction for a single input."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False
    )
    
    # Move to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
            
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # Decode only the generated part
    input_length = inputs['input_ids'].shape[1]
    generated_ids = outputs[0][input_length:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    return prediction


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_samples: List[Tuple[str, str]],
    max_new_tokens: int = 100
) -> Dict[str, Any]:
    """Evaluate model on test samples."""
    results = []
    total_metrics = {"cer": 0.0, "wer": 0.0}
    
    for i, (noisy_text, clean_text) in enumerate(tqdm(test_samples, desc="Evaluating")):
        try:
            prediction = generate_prediction(model, tokenizer, noisy_text, max_new_tokens)
            metrics = calculate_metrics(prediction, clean_text)
            
            # Accumulate metrics
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            # Store detailed result
            result = {
                "sample_id": i + 1,
                "input": noisy_text,
                "target": clean_text,
                "prediction": prediction,
                **metrics
            }
            results.append(result)
            
        except Exception as e:
            logging.error(f"Error processing sample {i+1}: {e}")
            continue
    
    # Calculate averages
    num_samples = len(results)
    if num_samples > 0:
        avg_metrics = {key: value / num_samples for key, value in total_metrics.items()}
    else:
        avg_metrics = {"cer": 0.0, "wer": 0.0}
    
    return {
        "summary": {
            "total_samples": num_samples,
            **avg_metrics
        },
        "detailed_results": results
    }


def save_results(results: Dict[str, Any], output_dir: str, config: Dict[str, Any]) -> None:
    """Save evaluation results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare complete output data
    output_data = {
        **results,
        "config": config
    }
    
    # Save JSON results
    json_path = output_path / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Save human-readable summary
    txt_path = output_path / "evaluation_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Model Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        
        summary = results["summary"]
        f.write(f"Total samples processed: {summary['total_samples']}\n")
        f.write(f"Average CER: {summary['cer']:.4f}\n")
        f.write(f"Average WER: {summary['wer']:.4f}\n\n")
        
        f.write("Sample Details:\n")
        f.write("-" * 50 + "\n")
        
        for result in results["detailed_results"]:
            f.write(f"\nSample {result['sample_id']}:\n")
            f.write(f"Input: {result['input']}\n")
            f.write(f"Target: {result['target']}\n")
            f.write(f"Prediction: {result['prediction']}\n")
            f.write(f"CER: {result['cer']:.4f}, WER: {result['wer']:.4f}\n")
            f.write("-" * 80 + "\n")
    
    logging.info(f"Results saved to {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model on OCR correction task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the fine-tuned model directory"
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=True, 
        help="Path to test dataset"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="evaluation_results", 
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=100, 
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None, 
        help="Maximum number of test samples to evaluate"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    logging.info("Starting model evaluation")
    
    try:
        # Load model and data
        model, tokenizer = load_model_and_tokenizer(args.model_path)
        test_samples = load_test_data(args.dataset_path, args.max_samples)
        
        # Run evaluation
        results = evaluate_model(model, tokenizer, test_samples, args.max_new_tokens)
        
        # Save results
        config = {
            "model_path": args.model_path,
            "dataset_path": args.dataset_path,
            "max_new_tokens": args.max_new_tokens,
            "max_samples": args.max_samples
        }
        save_results(results, args.output_dir, config)
        
        # Print summary
        summary = results["summary"]
        print(f"\n{'='*50}")
        print("EVALUATION COMPLETED")
        print(f"{'='*50}")
        print(f"Samples processed: {summary['total_samples']}")
        print(f"Average CER: {summary['cer']:.4f}")
        print(f"Average WER: {summary['wer']:.4f}")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()