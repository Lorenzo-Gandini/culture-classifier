
import json
from openai import OpenAI  
import requests
import os

from ftfy import fix_text
import unicodedata
import re

model_name = "deepseek/deepseek-r1-0528:free"
output_path_model = "deepseek" #serve per la cartella e file di output
translation_version = "v7"

PROMPT_TEMPLATE = """You are an expert OCR text restoration specialist. Your task is to fix OCR scanning errors while preserving the original text's integrity.

COMMON OCR ERRORS TO FIX:
• Character substitutions: O↔0, l↔1↔I, rn↔m, vv↔w, cl↔d, nn↔m, ci↔d
• Broken words: "th is" → "this", "a nd" → "and", "w ord" → "word"  
• Missing/extra spaces: "inthe" → "in the", "text .But" → "text. But"
• Quote artifacts: \" → " (fix escaped quotes to normal quotes)
• Case errors: random capitalization, missing capitals after periods
• Artifacts: remove stray | ~ ` symbols and Unicode remnants

PUNCTUATION PRESERVATION:
• Keep em-dashes (—) as they are typical of 19th century literature
• Fix only quote encoding errors: \"text\" → "text"
• Preserve original dialogue formatting and complex punctuation patterns

CORRECTION EXAMPLES:
"Th e qu ick br0wn f0x jvmps 0ver the |azy d0g." → "The quick brown fox jumps over the lazy dog."
"lt was a dark st0rmy night.Sudcenly a sh0t rang 0ut!" → "It was a dark stormy night. Suddenly a shot rang out!"
"The p0em 0f \"Thalaba,\" the vampyre c0rse" → "The poem of "Thalaba," the vampyre corse"
"affecti0n.—A supp0siti0n alluded t0" → "affection.—A supposition alluded to"

STRICT RULES:
1. Fix ONLY obvious OCR errors - do NOT modernize, paraphrase, or interpret
2. Preserve original grammar, vocabulary, and sentence structure exactly
3. Maintain paragraph breaks and text formatting
4. Keep em-dashes (—) and period-style literary punctuation intact
5. Fix quote encoding (\" → ") but preserve quote placement and dialogue structure
6. If uncertain about a correction, leave the text unchanged
7. Output ONLY the corrected text with no explanations

OCR text:
{input_text}

Cleaned text:
"""

def pre_clean_text(text):
    # fix Unicode errors (quotes, dashes, accents)
    text = fix_text(text)

    # normalize Unicode
    text = unicodedata.normalize("NFKD", text)
    
    # fix common OCR artifacts before LLM processing. TOCHECK: Maybe some of them must be keeped ?
    ocr_replacements = {
        # Ligatures comuni
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
        # Quote standardization
        ''': "'", ''': "'", '"': '"', '"': '"',
        '«': '"', '»': '"',
        # Dashes
        '–': '-', '—': '-', '…': '...',
    }
    
    for old, new in ocr_replacements.items():
        text = text.replace(old, new)

    # remove control characters but keep the line breaks
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # clean up garbage patterns
    text = re.sub(r"[^\w\s.,;:'\"!?()\[\]\-]", "", text)

    # normalize spacing 
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # Max 2 consecutive newlines

    return text.strip()

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

def post_process_final_text(text):
    """Post-processing per migliorare la coerenza finale"""
    #normalize spaces after punctuation
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    text = re.sub(r'([,;:])\s*([a-zA-Z])', r'\1 \2', text)
    
    #fix spaces before punctuation
    text = re.sub(r'\s+([.!?,;:])', r'\1', text)
    
    # Normalize line breaks
    text = re.sub(r'\n\n\n+', '\n\n', text)
    
    #Capitalize after final dots.
    text = re.sub(r'(\. )([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    
    return text.strip()


with open("data/eng/the_vampyre_ocr.json", "r") as f_ocr:
    ocr_data = json.load(f_ocr)

max_token_input = 2000
first_key = list(ocr_data.keys())[0]
first_text = ocr_data[first_key]

print(f"📖 Elaborating : {first_key}")
print(f"📏 Original length: {len(first_text)} chars")

first_text = pre_clean_text(first_text) 
print(f"📏 Length after pre-processing: {len(first_text)} chars")

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
final_text = post_process_final_text(final_text)

# Salva il risultato. Aggiungo più elementi per il debug e la tracciabilità
output = {
    "id": first_key,
    "original_ocr_text": ocr_data[first_key],
    "preprocessed_text": first_text,
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