import torch
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from ocr_dataset import OCRDataset, retrieve_datasets
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator 
import deepspeed 
import os 
import editdistance

def main():

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to test data"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--deepspeed_config_path",
        type=str,
        default="deepspeed_inference.json",
        help="Path to the DeepSpeed config file"
    )   
    args = parser.parse_args()
    
    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"  # Automatically handle GPU placement
    )

    train_data, test_data = retrieve_datasets(args.dataset_path, test_size=0.5)

    # Create pipeline - much simpler!
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Rest of your code stays the same...
    def cer(s1, s2):
        return editdistance.eval(s1, s2) / max(len(s2), 1)

    def wer(s1, s2):
        return editdistance.eval(s1.split(), s2.split()) / max(len(s2.split()), 1)

    # Inference loop
    total_cer = 0
    total_wer = 0
    n = 0

    output_path = os.path.join(args.output_path, "inference.txt")
    with torch.no_grad(), open(output_path, "w", encoding="utf-8") as f:
        for (noisy, clean) in tqdm(test_data):
            
            # Add generation parameters for better control
            output = pipe(
                noisy, 
                max_new_tokens=512, 
                do_sample=False,  # Deterministic generation
                pad_token_id=tokenizer.eos_token_id,  # Handle padding
                return_full_text=False  # Only return generated part, not input
            )
            output = output[0]["generated_text"]
            
            total_cer += cer(output, clean)
            total_wer += wer(output, clean)
            n += 1
            
            f.write(f"Sample {n}:\n")
            f.write(f"Ground truth: {clean}\n")
            f.write(f"Prediction  : {output}\n")
            f.write(f"CER: {cer(output, clean):.3f}, WER: {wer(output, clean):.3f}\n\n")
        
        avg_cer = total_cer / n
        avg_wer = total_wer / n
        
        f.write(f"\n✅ Average CER: {avg_cer:.3f}\n")
        f.write(f"✅ Average WER: {avg_wer:.3f}\n")

if __name__ == "__main__":
    main()