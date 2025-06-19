import json 
import argparse
import os
import random 

def parse_args():

    parser = argparse.ArgumentParser(prog="Dataset Preprocess", usage='%(prog)s [options]')
    parser.add_argument("--data_path", type=str, required=True, help="Path to the folder containing BOTH clean.json and noisy.json")
    parser.add_argument("--prompts_path", type=str, default="./prompts_preprocess/", help="Path to the preprocess prompts folder")

    return parser.parse_args()

def open_json(path):

    """
        Function that opens and returns the json files

        Args: 
            - path: Directory containing clean and noisy files
        Returns: 
            - clean: clean text
            - noisy: noisy text 
    """
    clean_path = os.path.join(path, "clean.json")
    noisy_path = os.path.join(path, "noisy.json")

    with open(clean_path, "r+") as f:
        clean = json.load(f)
    
    with open(noisy_path, "r+") as f:
        noisy = json.load(f)

    return clean, noisy  

def load_input_prompts(text_lang, base_folder='prompts'):

    """
    Loads all prompts for all combinations of text language and prompt language,
    separated by prefixes and suffixes.
    
    Returns nested dict:
    {text_lang: {prompt_lang: {"prefixes": [...], "suffixes": [...]}}}
    """

    text_languages = [text_lang]
    prompt_languages = ["eng", "ita"]
    
    all_prompts = {}

    for text_lang in text_languages:


        all_prompts[text_lang] = {}

        for prompt_lang in prompt_languages:

            folder_name = f"{prompt_lang}_prompt_{text_lang}_text"
            folder_path = os.path.join(base_folder, folder_name)
            
            prefixes = []
            suffixes = []
            
            if not os.path.exists(folder_path):
                print(f"Folder {folder_path} does not exist, skipping.")
                all_prompts[text_lang][prompt_lang] = {"prefixes": [], "suffixes": []}
                continue
            
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                if filename.startswith("prefix"):
                    prefixes.extend(lines)
                elif filename.startswith("suffix"):
                    suffixes.extend(lines)
            
            all_prompts[text_lang][prompt_lang] = {
                "prefixes": prefixes,
                "suffixes": suffixes
            }
    
    return all_prompts

def load_response_prompts(text_lang, base_folder='prompts/'):

    
    """
    Loads all response prompts for all combinations of text language and prompt language.

    Returns nested dict:
    {text_lang: {prompt_lang: [prompts]}}
    """

    prompt_languages = ["eng", "ita"]
    text_languages = [text_lang]
    
    base_folder = os.path.join(base_folder, "response_prompts")

    all_response_prompts = {}

    
    for text_lang in text_languages:

        all_response_prompts[text_lang] = {}
        
        for prompt_lang in prompt_languages:

            folder_name = f"{prompt_lang}_prompt_{text_lang}_text"
            folder_path = os.path.join(base_folder, folder_name)

            prompts = []
            
            if not os.path.exists(folder_path):
                print(f"Folder {folder_path} does not exist, skipping.")
                all_response_prompts[text_lang][prompt_lang] = []
                continue
            
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    prompts.extend(lines)
            
            all_response_prompts[text_lang][prompt_lang] = prompts
    
    return all_response_prompts


def generate_augmented_pairs(noisy_texts, clean_texts, input_prompts, response_prompts, text_lang, seed=None):
    """
    Generate augmented input-response pairs.

    Args:
      - noisy_texts: dict of {key: noisy_text} (OCR text)
      - clean_texts: dict of {key: clean_text} (clean reference)
      - input_prompts: nested dict with structure
          input_prompts[text_lang][prompt_lang]['prefixes'] and ['suffixes']
      - response_prompts: nested dict with structure
          response_prompts[text_lang][prompt_lang]
      - text_lang: str, language of the text, e.g. 'eng' or 'ita'
      - seed: int, optional for reproducibility

    Returns:
      - augmented_inputs: dict {key: augmented_input_text}
      - augmented_responses: dict {key: augmented_response_text}
    """
    
    if seed is not None:
        random.seed(seed)

    augmented_inputs = {}
    augmented_responses = {}

    prompt_languages = list(input_prompts[text_lang].keys())

    for key, noisy_text in noisy_texts.items():
        # 1. Randomly pick prompt language for input prompt
        prompt_lang = random.choice(prompt_languages)

        # 2. Randomly pick prefix or suffix
        use_prefix = random.choice([True, False])

        prefixes = input_prompts[text_lang][prompt_lang].get('prefixes', [])
        suffixes = input_prompts[text_lang][prompt_lang].get('suffixes', [])

        if use_prefix and prefixes:
            input_prompt = random.choice(prefixes)
            augmented_input = f"{input_prompt.strip()}\n{noisy_text.strip()}"
        elif suffixes:
            input_prompt = random.choice(suffixes)
            augmented_input = f"{noisy_text.strip()}\n{input_prompt.strip()}"
        else:
            # fallback if no prefix or suffix available
            augmented_input = noisy_text.strip()

        # 3. For response, pick prompt from response_prompts with same prompt_lang
        resp_prompts_for_lang = response_prompts[text_lang].get(prompt_lang, [])
        if resp_prompts_for_lang:
            response_prompt = random.choice(resp_prompts_for_lang).strip()
            clean_text = clean_texts[key].strip()
            augmented_response = f"{response_prompt}\n{clean_text}"
        else:
            augmented_response = clean_texts[key].strip()

        augmented_inputs[key] = augmented_input
        augmented_responses[key] = augmented_response

    return augmented_inputs, augmented_responses



def save_files(responses, inputs, path):

    noisy_path = os.path.join(path, "noisy.json")
    clean_path = os.path.join(path, "clean.json")

    
    with open(noisy_path, "w+") as f:
        print(inputs)
        json.dump(inputs, f)
    
    with open(clean_path, "w+") as f:
        json.dump(responses, f)



if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path 
    prompts_path = args.prompts_path 

    # File is a dictionary made of 
    #    index: text 
    # e.g. 1: "the quick brown fox" 

    clean, noisy = open_json(data_path)
    
    if("eng" in data_path):
        text_lang = "eng"
    else:
        text_lang = "ita"

    input_prompts = load_input_prompts(text_lang, prompts_path)
    response_prompts = load_response_prompts(text_lang, prompts_path)

    # Create folder for preprocessed text 
    save_path = "./data_preprocessed"
    save_path = os.path.join(save_path, f"{text_lang}")
    os.makedirs(save_path, exist_ok=True)
    

    augmented_inputs, augmented_responses = generate_augmented_pairs(noisy, clean, input_prompts, response_prompts, text_lang, 42)
    

    save_files(responses=augmented_responses,
                inputs=augmented_inputs,
                path=save_path)
    




