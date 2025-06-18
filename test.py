import json
import random
import time
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from rouge import Rouge
import requests
import os
import argparse

# Configuration
OCR_PATH = "ocr_datasets/eng/the_vampyre_ocr.json"
HUMAN_CLEANED_PATH = "ocr_datasets/eng/the_vampyre_clean.json"
GROUP_NAME = "SocioEmbeddings"
API_KEY = ""

CORRECTION_MODELS = [
    "meta-llama/llama-3.3-8b-instruct:free",
    "deepseek/deepseek-r1-0528"
]
JUDGE_MODEL = "meta-llama/llama-3.3-8b-instruct:free"

PROMPT_TEMPLATE = """You are a text restoration specialist. Your task is to fix OCR scanning errors in historical text while preserving the original content exactly as intended by the author.

COMMON OCR ERRORS TO FIX:
• Character confusion: O↔0, l↔1↔I, rn↔m, vv↔w, cl↔d, nn↔n, u↔n
• Word breaks: "th is" → "this", "a nd" → "and", "w hen" → "when"
• Missing spaces: "andthe" → "and the", "inthe" → "in the"
• Common substitutions: "tbe" → "the", "lhe" → "the", "af" → "of", "arid" → "and"
• Punctuation errors: Missing periods, comma/period confusion, quote mark errors

STRICT PRESERVATION RULES:
1. Fix ONLY obvious scanning errors - do not modernize, paraphrase, or improve the writing
2. Keep original punctuation style (dashes, quotation marks, capitalization patterns)
3. Maintain the author's vocabulary, sentence structure, and writing style
4. Do not add explanatory text, footnotes, or editorial comments
5. If uncertain about a correction, leave the text as-is rather than guess
6. Mantain unicode characters if used and don't look like an ocr mistake (\u2014 → \u2014)

CONTEXT CLUES:
• Use surrounding words to disambiguate unclear characters
• Consider the time period and writing conventions of the source
• Grammar and meaning should guide corrections, but respect historical language patterns

TEXT TO CORRECT:
{input_text}

CORRECTED TEXT (return only the corrected text without additional commentary, explanations, or formatting changes):"""

JUDGE_PROMPT = """You are evaluating the quality of text correction. Compare the original corrupted text with its corrected version and select the most appropriate rating word.

Consider these aspects in order of importance:

**FIRST - Accuracy**: Were the obvious errors fixed?
- Spelling mistakes corrected
- Character recognition errors resolved (O/0, l/1/I, rn/m, etc.)
- Broken word boundaries fixed ("th is" → "this")
- Punctuation restored where clearly needed

**SECOND - Preservation**: Was the original content maintained?
- No meaning changes or rewording
- Historical language style kept intact
- Original sentence structure preserved
- No content added or removed
- Mantained unicode characters if used and don't look like an ocr mistake (\u2014 → \u2014)

**THIRD - Readability and Completeness**:
- Text flows naturally and makes sense
- All obvious errors were addressed
- No new errors were introduced

RATING SCALE (choose exactly one word, listed from best to worst):

**EXCELLENT**: Nearly flawless correction with perfect preservation of original content
**GOOD**: Minor issues remain, but highly readable and accurate overall  
**ADEQUATE**: Readable with some remaining errors or slight preservation issues
**POOR**: Significant problems affecting readability or meaning
**TERRIBLE**: Major errors, unreadable sections, or substantial content changes

EXAMPLES:
- Good response: EXCELLENT
- Bad response: The text is excellent because... EXCELLENT
- Bad response: 5 EXCELLENT

TEXTS TO EVALUATE:

Original corrupted text:
{ocr_text}

Corrected text:
{cleaned_text}

Your rating (one word only, no explanations, no numbers):"""


def load_data(sample_size=25):
    """Load OCR and human-cleaned datasets"""
    with open(OCR_PATH, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)
    with open(HUMAN_CLEANED_PATH, 'r', encoding='utf-8') as f:
        human_data = json.load(f)
    
    # Select sample sentences
    keys = sorted(ocr_data.keys(), key=int)[:sample_size]
    return {k: ocr_data[k] for k in keys}, {k: human_data[k] for k in keys}

def call_llm_api(prompt, model_name, max_retries=3):
    """Call OpenRouter API with robust error handling"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.01,
        "max_tokens": 2000
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                print(f"API Error ({response.status_code}): {response.text}")
        except Exception as e:
            print(f"Request failed (attempt {attempt+1}): {str(e)}")
        
        time.sleep(2**attempt)  # Exponential backoff
    
    print(f"Failed after {max_retries} attempts")
    return None

# STEP 1 OCR Correction
def run_ocr_correction(sample_size=25):
    """Step 1: Clean text with multiple LLMs"""
    ocr_data, human_data = load_data(sample_size)
    print(f"Loaded {len(ocr_data)} sentences")
    
    cleaned_results = {}
    for model in CORRECTION_MODELS:
        model_name = model.split("/")[-1].split(":")[0]
        cleaned = {}
        print(f"Cleaning with {model_name}...")
        
        for key, text in ocr_data.items():
            time.sleep(5)
            prompt = PROMPT_TEMPLATE.format(input_text=text)
            cleaned_text = call_llm_api(prompt, model)
            cleaned[key] = cleaned_text if cleaned_text else "CLEANING_FAILED"
            print(f"Sentence {key} cleaned")
        
        cleaned_results[model_name] = cleaned
        
        # Save cleaned output
        output_path = f"{GROUP_NAME}-hw2_ocr-{model_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)
        print(f"Saved cleaned text to {output_path}")
    
    return cleaned_results

# STEP 2 LLM-as-Judge Evaluation
def run_llm_judge_evaluation():
    """Step 2 Evaluate cleaned text with LLM-as-Judge"""
    ocr_data, _ = load_data()
    
    # Load all cleaned outputs
    cleaned_results = {}
    for model in CORRECTION_MODELS:
        model_name = model.split("/")[-1].split(":")[0]
        input_path = f"{GROUP_NAME}-hw2_ocr-{model_name}.json"
        with open(input_path, 'r', encoding='utf-8') as f:
            cleaned_results[model_name] = json.load(f)
    
    # Evaluate each model
    judge_results = {}
    for model_name, cleaned_data in cleaned_results.items():
        model_ratings = {}
        print(f"\nEvaluating {model_name} with {JUDGE_MODEL}...")
        
        for key, text in cleaned_data.items():
            if text == "CLEANING_FAILED":
                model_ratings[key] = None
                continue
                
            prompt = JUDGE_PROMPT.format(
                ocr_text=ocr_data[key],
                cleaned_text=text
            )
            time.sleep(10)
            response = call_llm_api(prompt, JUDGE_MODEL)
            
            # Map word responses to numbers
            rating_map = {
                "EXCELLENT": 5,
                "GOOD": 4,
                "ADEQUATE": 3, 
                "POOR": 2,
                "TERRIBLE": 1
            }
            
            if response:
                clean_response = response.strip().upper()
                for word, score in rating_map.items():
                    if word in clean_response:
                        model_ratings[key] = score
                        break
                else:
                    model_ratings[key] = None
            else:
                model_ratings[key] = None
        
        judge_results[model_name] = model_ratings
    
    # Save judge outputs
    judge_path = f"{GROUP_NAME}-hw2_ocr-judge.json"
    with open(judge_path, "w", encoding="utf-8") as f:
        json.dump(judge_results, f, indent=2, ensure_ascii=False)
    print(f"Saved judge evaluations to {judge_path}")
    
    return judge_results

# STEP 3 Metrics Calculation
def run_metrics(human_ratings_path=None):
    """Step 3 Calculate metrics with human ratings"""
    # Load datasets
    ocr_data, human_data = load_data()
    
    # Load cleaned outputs
    cleaned_results = {}
    for model in CORRECTION_MODELS:
        model_name = model.split("/")[-1].split(":")[0]
        input_path = f"{GROUP_NAME}-hw2_ocr-{model_name}.json"
        with open(input_path, 'r', encoding='utf-8') as f:
            cleaned_results[model_name] = json.load(f)
    
    # Load judge evaluations
    judge_path = f"{GROUP_NAME}-hw2_ocr-judge.json"
    with open(judge_path, 'r', encoding='utf-8') as f:
        judge_results = json.load(f)
    
    # Load or collect human ratings
    if human_ratings_path and os.path.exists(human_ratings_path):
        with open(human_ratings_path, 'r', encoding='utf-8') as f:
            human_ratings = json.load(f)
        print("Loaded existing human ratings")
    else:
        human_ratings = {}
        print("Collecting Human Ratings (simulated)...")
        for model_name in cleaned_results.keys():
            ratings = {}
            for key in cleaned_results[model_name].keys():
                # In real implementation, show humans the texts
                ratings[key] = random.randint(4, 5)  # Simulated rating
            human_ratings[model_name] = ratings
        
        # Save human ratings
        hr_path = human_ratings_path or f"{GROUP_NAME}-hw2_ocr-human_ratings.json"
        with open(hr_path, 'w', encoding='utf-8') as f:
            json.dump(human_ratings, f, indent=2, ensure_ascii=False)
        print(f"Saved human ratings to {hr_path}")
    
    # Calculate ROUGE scores
    rouge = Rouge()
    metrics_results = {}
    
    for model_name in cleaned_results.keys():
        model_metrics = []
        for key in cleaned_results[model_name].keys():
            if cleaned_results[model_name][key] == "CLEANING_FAILED":
                continue
                
            try:
                scores = rouge.get_scores(
                    cleaned_results[model_name][key], 
                    human_data[key]
                )[0]
            except:
                scores = {"rouge-1": {"f": 0}, "rouge-2": {"f": 0}, "rouge-l": {"f": 0}}
            
            metrics_entry = {
                "sentence_id": key,
                "rouge1": scores["rouge-1"]["f"],
                "rouge2": scores["rouge-2"]["f"],
                "rougeL": scores["rouge-l"]["f"],
                "llm_judge": judge_results[model_name].get(key, None),
                "human_rating": human_ratings[model_name].get(key, None)
            }
            model_metrics.append(metrics_entry)
        
        metrics_results[model_name] = model_metrics
    
    # Calculate correlations
    correlation_results = {}
    for model_name, metrics in metrics_results.items():
        df = pd.DataFrame(metrics)
        df = df.dropna()  # Remove incomplete entries
        
        if len(df) < 5:
            print(f"Not enough data for {model_name} correlation analysis")
            continue
        
        correlations = {
            "rouge1_vs_human": spearmanr(df["rouge1"], df["human_rating"]).correlation,
            "rouge2_vs_human": spearmanr(df["rouge2"], df["human_rating"]).correlation,
            "rougeL_vs_human": spearmanr(df["rougeL"], df["human_rating"]).correlation,
            "llm_judge_vs_human": spearmanr(df["llm_judge"], df["human_rating"]).correlation,
            "rouge1_vs_llm_judge": spearmanr(df["rouge1"], df["llm_judge"]).correlation,
            "rouge2_vs_llm_judge": spearmanr(df["rouge2"], df["llm_judge"]).correlation,
            "rougeL_vs_llm_judge": spearmanr(df["rougeL"], df["llm_judge"]).correlation
        }
        correlation_results[model_name] = correlations
    
    # Save full results
    metrics_path = f"{GROUP_NAME}-hw2_ocr-metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metrics": metrics_results,
            "correlations": correlation_results
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved full metrics to {metrics_path}")
    
    # Print correlation summary
    print("Correlation Results:")
    for model, corrs in correlation_results.items():
        print(f"Model: {model}")
        print(f"ROUGE-1 vs Human: {corrs['rouge1_vs_human']:.3f}")
        print(f"ROUGE-2 vs Human: {corrs['rouge2_vs_human']:.3f}")
        print(f"ROUGE-L vs Human: {corrs['rougeL_vs_human']:.3f}")
        print(f"LLM-as-Judge vs Human: {corrs['llm_judge_vs_human']:.3f}")
        print(f"ROUGE-1 vs LLM-as-Judge: {corrs['rouge1_vs_llm_judge']:.3f}")
        print(f"ROUGE-2 vs LLM-as-Judge: {corrs['rouge2_vs_llm_judge']:.3f}")
        print(f"ROUGE-L vs LLM-as-Judge: {corrs['rougeL_vs_llm_judge']:.3f}")
    
    return metrics_results, correlation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Correction Pipeline")
    parser.add_argument("--step", type=int, choices=[1, 2, 3], required=True,
                        help="Pipeline step to execute: 1=OCR Correction, 2=LLM Judge, 3=Metrics")
    parser.add_argument("--human-ratings", type=str, 
                        help="Path to human ratings JSON file (for step 3)")
    parser.add_argument("--sample-size", type=int, default=25,
                        help="Number of sentences to process")
    
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    
    if args.step == 1:
        print("Running Step 1: OCR Correction")
        run_ocr_correction(args.sample_size)
        print("Step 1 Complete!")
    
    elif args.step == 2:
        print("Running Step 2: LLM-as-Judge Evaluation")
        run_llm_judge_evaluation()
        print("Step 2 Complete!")
    
    elif args.step == 3:
        print("Running Step 3: Metrics Calculation")
        run_metrics(args.human_ratings)
        print("Step 3 Complete!")