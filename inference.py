import torch
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from ocr_dataset import retrieve_datasets
import editdistance
import time
import os 

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--output_path", type=str, default="evaluation_results.json", help="Path to save evaluation results")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--test_samples", type=int, default=None, help="Limit test samples for debugging")
    
    args = parser.parse_args()
    
    print("🚀 Starting evaluation...")
    
    # Load tokenizer and model
    print("📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("🤖 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,  # Use float16 for speed
    )
    model.eval()
    
    print("📊 Loading dataset...")
    _, test_data = retrieve_datasets(args.dataset_path, test_size=0.5)
    test_samples = list(test_data)
    
    if args.test_samples:
        test_samples = test_samples[:args.test_samples]
    
    print(f"📈 Processing {len(test_samples)} samples...")
    
    # Metrics
    def calculate_cer(pred, target):
        return editdistance.eval(pred, target) / max(len(target), 1)
    
    def calculate_wer(pred, target):
        pred_words = pred.split()
        target_words = target.split()
        return editdistance.eval(pred_words, target_words) / max(len(target_words), 1)
    
    # Results storage
    results = []
    total_cer = 0
    total_wer = 0
    
    # Process samples one by one (most reliable approach)
    start_time = time.time()
    
    with open("results.txt", "w", encoding="utf-8") as f:
        for i, (noisy_text, clean_text) in enumerate(tqdm(test_samples, desc="Evaluating")):
            try:
                # Tokenize input
                inputs = tokenizer(
                    noisy_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=False
                )
                
                # Move to device
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        num_beams=1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True
                    )
                
                # Decode only the generated part
                input_length = inputs['input_ids'].shape[1]
                generated_ids = outputs[0][input_length:]
                prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                # Calculate metrics
                cer_score = calculate_cer(prediction, clean_text)
                wer_score = calculate_wer(prediction, clean_text)
                
                total_cer += cer_score
                total_wer += wer_score
                
                # Store result
                result = {
                    "sample_id": i + 1,
                    "input": noisy_text,
                    "target": clean_text,
                    "prediction": prediction,
                    "cer": cer_score,
                    "wer": wer_score
                }
                results.append(result)
                
                # Write to file
                f.write(f"Sample {i+1}:\n")
                f.write(f"Input: {noisy_text}\n")
                f.write(f"Target: {clean_text}\n")
                f.write(f"Prediction: {prediction}\n")
                f.write(f"CER: {cer_score:.4f}, WER: {wer_score:.4f}\n")
                f.write("-" * 80 + "\n")
                
                # Progress update every 10 samples
                if (i + 1) % 10 == 0:
                    avg_cer = total_cer / (i + 1)
                    avg_wer = total_wer / (i + 1)
                    elapsed = time.time() - start_time
                    samples_per_sec = (i + 1) / elapsed
                    print(f"Progress: {i+1}/{len(test_samples)} | CER: {avg_cer:.4f} | WER: {avg_wer:.4f} | Speed: {samples_per_sec:.2f} samples/sec")
                    f.flush()  # Force write
                
            except Exception as e:
                print(f"Error processing sample {i+1}: {str(e)}")
                print(f"Input length: {len(noisy_text)}")
                continue
    
    # Final calculations
    num_samples = len(results)
    if num_samples > 0:
        final_cer = total_cer / num_samples
        final_wer = total_wer / num_samples
        
        print(f"\n🎯 Final Results:")
        print(f"Samples processed: {num_samples}")
        print(f"Average CER: {final_cer:.4f}")
        print(f"Average WER: {final_wer:.4f}")
        
        total_time = time.time() - start_time
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per sample: {total_time/num_samples:.2f} seconds")
        
        # Save detailed results
        output_data = {
            "summary": {
                "total_samples": num_samples,
                "average_cer": final_cer,
                "average_wer": final_wer,
                "total_time_seconds": total_time,
                "samples_per_second": num_samples / total_time
            },
            "config": {
                "model_path": args.model_path,
                "dataset_path": args.dataset_path,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size
            },
            "detailed_results": results
        }
        
        output_path = os.path.join(args.output_path, "inference.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {args.output_path}")
        
    else:
        print("❌ No samples were processed successfully!")

if __name__ == "__main__":
    main()