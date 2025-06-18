import os
import torch
import logging
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer
)
from accelerate import Accelerator
from ocr_dataset import retrieve_datasets, OCRDataset

os.environ["WANDB_DISABLED"] = "true"

def main(model_path, dataset_path, output_file):
    print("Loading pre-trained model for DeepSpeed training")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model normally - DeepSpeed will handle distribution
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        # Don't use device_map with DeepSpeed
        # device_map=None,
    )
    
    train_data, test_data = retrieve_datasets(path=dataset_path)
    train_dataset = OCRDataset(tokenizer=tokenizer, data=train_data)
    test_dataset = OCRDataset(tokenizer=tokenizer, data=test_data)
    
    os.makedirs(output_file, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_file,
        per_device_train_batch_size=2,  # Can be larger with DeepSpeed
        gradient_accumulation_steps=4,
        num_train_epochs=20,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=False,
        save_steps=500,
        save_strategy="no", 
        logging_steps=50,
        learning_rate=3e-4,
        warmup_steps=2000,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        run_name="minerva-ocr-ft",
        push_to_hub=False,
        log_level="info",
        bf16=True,
        # DeepSpeed configuration
        deepspeed="deepspeed_config.json",  # Path to DeepSpeed config
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
    )
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger()
    
    logger.info("Starting DeepSpeed training...")
    trainer.train()

    metrics = trainer.evaluate()

    # Optionally print metrics
    print(metrics)

    trainer.save_model(output_file)
    tokenizer.save_pretrained(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Minerva model with DeepSpeed")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained model directory or model name"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where to save the file"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data_preprocessed/eng",
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    args = parser.parse_args()
    
    print(f"Using dataset {args.dataset_path}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    main(args.model_path, args.dataset_path, args.output_path)