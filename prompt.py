
import json
from openai import OpenAI  
import requests
import os
import re

model_name = "deepseek/deepseek-r1-0528:free"
# model_name = "deepseek/deepseek-chat-v3-0324:free"
       
# model_name = "microsoft/phi-4-reasoning-plus:free"

# model_name = "google/gemma-3-27b-it:free"
# model_name = "google/gemini-2.0-flash-exp:free"

# model_name = "opengvlab/internvl3-14b:free"

# model_name = "meta-llama/llama-3.3-8b-instruct:free"
# model_name = "meta-llama/llama-4-maverick:free" 
# model_name = "meta-llama/llama-4-scout:free"

# model_name = "qwen/qwen-2.5-coder-32b-instruct:free"
# model_name = "qwen/qwen-2.5-72b-instruct:free"
# model_name = "qwen/qwen-2.5-7b-instruct:free"
# model_name = "qwen/qwen-2.5-vl-7b-instruct:free"

# model_name = "nousresearch/deephermes-3-mistral-24b-preview:free" 
# model_name = "mistralai/mistral-nemo:free"
# model_name = "cognitivecomputations/dolphin3.0-mistral-24b:free"
# model_name = "mistralai/mistralai/devstral-small:free"

output_path_model = model_name.split("/")[1].split(":")[0]
translation_version = "v2"

PROMPT_TEMPLATE = """You are an expert OCR text restoration specialist. Your task is to fix OCR scanning errors while preserving the original text's integrity.

STEP 1 - PRE-PROCESSING FIXES:
• Fix Unicode artifacts and ligatures: ﬁ→fi, ﬂ→fl, ﬀ→ff, ﬃ→ffi, ﬄ→ffl
• Normalize quotes: ''→', ""→", «»→""
• Fix escaped quotes: \"→" (but preserve em-dashes —)
• Remove control characters and garbage: |~` and similar artifacts
• Normalize horizontal spacing: multiple spaces/tabs → single space
• Limit consecutive line breaks to maximum 2

STEP 2 - MAIN OCR CORRECTIONS:
• Character substitutions: O↔0, l↔1↔I, rn↔m, vv↔w, cl↔d, nn↔m, ci↔d
• Broken words: "th is"→"this", "a nd"→"and", "w ord"→"word", "0f"→"of"
• Missing/extra spaces: "inthe"→"in the", "text.But"→"text. But"
• Case errors: random capitalization, missing capitals after periods
• Common word patterns: "0r"→"or", "f0r"→"for", "t0"→"to", "h0use"→"house"

STEP 3 - POST-PROCESSING REFINEMENT:
• Ensure single space after punctuation: "word.Another"→"word. Another"
• Fix spaces before punctuation: "word ."→"word."
• Capitalize after sentence endings: ". word"→". Word"
• Normalize excessive line breaks: 3+ newlines → 2 newlines maximum

LITERARY TEXT PRESERVATION:
• Keep em-dashes (—) intact - typical of 19th century literature
• Preserve dialogue formatting and complex punctuation patterns
• Maintain original paragraph structure and line breaks
• Keep archaic spelling if not clearly OCR error (e.g., "colour", "honour")

CORRECTION EXAMPLES:
Input: "Th e qu ick br0wn f0x jvmps 0ver the |azy d0g ."
Output: "The quick brown fox jumps over the lazy dog."

Input: "lt was a dark st0rmy night.sudcenly a sh0t rang 0ut !"
Output: "It was a dark stormy night. Suddenly a shot rang out!"

Input: "The p0em 0f \"Thalaba ,\" the vampyre c0rse 0f"
Output: "The poem of "Thalaba," the vampyre corse of"

Input: "affecti0n.—A supp0siti0n alluded t0 in the text .but"
Output: "affection.—A supposition alluded to in the text. But"

Input: "he cried ,\"again baffled !\" t0 which a l0ud laugh"
Output: "he cried, "Again baffled!" to which a loud laugh"

PROCESSING STEPS TO FOLLOW:
1. Apply pre-processing fixes (Unicode, ligatures, spacing)
2. Correct OCR character errors and broken words
3. Apply post-processing refinement (punctuation spacing, capitalization)
4. Preserve literary formatting (em-dashes, dialogue structure)

STRICT RULES:
• Fix ONLY obvious OCR errors - do NOT modernize, paraphrase, or interpret
• Preserve original grammar, vocabulary, and sentence structure exactly
• Maintain paragraph breaks and text formatting
• If uncertain about a correction, leave the text unchanged
• Output ONLY the corrected text with no explanations or formatting

OCR text:
{input_text}

Cleaned text:
"""

def load_api_key(path="key.txt"):
    with open(path, "r") as f:
        return f.read().strip()

API_KEY = load_api_key()

def estimate_tokens(text, chars_per_token=4):
    return len(text) // chars_per_token

def split_text_by_paragraphs(text, max_tokens, chars_per_token=4):
    #Now the chunk respect also the paragraphs,
    max_chars = max_tokens * chars_per_token
    chunks = []
    
    paragraphs = text.split('\n\n')     #split in paragraphs
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If this paragraph is added, the total length will exceed max_chars?    
        test_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
        
        if len(test_chunk) <= max_chars:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If the single par is too long, split it into sentences
            if len(para) > max_chars:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_chunk = ""
                
                for sent in sentences:
                    if len(f"{temp_chunk} {sent}".strip()) <= max_chars:
                        temp_chunk = f"{temp_chunk} {sent}".strip()
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = sent
                
                if temp_chunk:
                    current_chunk = temp_chunk
                else:
                    current_chunk = ""
            else:
                current_chunk = para
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


with open("data/eng/the_vampyre_ocr.json", "r") as f_ocr:
    ocr_data = json.load(f_ocr)

max_token_input = 2000
first_key = list(ocr_data.keys())[0]
first_text = ocr_data[first_key]

print("\n ====================== \n🔍 Starting text cleaning process")
print("🔧 Model used for processing:", model_name)
print(f"📖 Elaborating text with ID: {first_key} ")

chunks = split_text_by_paragraphs(first_text, max_tokens=max_token_input, chars_per_token=4)
print(f"🔪 Text split in {len(chunks)} chunks")

# Processing chunks (mantiene la tua logica originale)
cleaned_chunks = []

for idx, chunk in enumerate(chunks):
    prompt = PROMPT_TEMPLATE.format(input_text=chunk.strip())
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1 #0.3. Low value increase consistency ?
        }
    )

    # Debug output to see also resoning and other stuff from the model
    print("Raw JSON response:")
    print(json.dumps(response.json(), indent=2))

    result = response.json()["choices"][0]["message"]["content"].strip()
    cleaned_chunks.append(result)
    print(f"✔️ Chunk {idx + 1}/{len(chunks)} completato")


final_text = "\n\n".join(cleaned_chunks)

# Salva il risultato. Aggiungo più elementi per il debug e la tracciabilità
output = {
    "id": first_key,
    "original_ocr_text": ocr_data[first_key],
    "cleaned_text": final_text,
    "processing_info": {
        "chunks_count": len(chunks),
        "model_used": model_name
    }
}


output_path = f"results/{output_path_model}/{output_path_model}_cleaned_optimized_{translation_version}.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ Elaboration completed!")
print(f"📁 Saved in : {output_path}")