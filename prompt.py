
import json
from openai import OpenAI  
import requests
import os

from ftfy import fix_text
import unicodedata
import re

model_name = "deepseek/deepseek-r1-0528:free"

PROMPT_TEMPLATE = """
You are a highly precise text restoration model specialized in correcting text that has been corrupted by OCR scanning errors.

You are given a passage that contains **typical OCR errors**, such as:
- broken words or missing spaces
- misrecognized letters or digits (e.g. "0" → "O", "1" → "l", "rn" → "m", "vv" → "w")
- unwanted characters, Unicode artifacts, or symbols (e.g. "\u017f", smart quotes)
- misspellings that clearly result from OCR noise (e.g. "corrvpl" → "corrupt", "haue" → "have")
- false capitalization or inconsistent casing
- missing or incorrect punctuation (e.g. “,” → ",", random hyphens or em dashes)

You MUST:
- Correct only what is clearly due to OCR or typographic noise.
- **Preserve the original structure, style, and grammar**.
- Do **NOT modernize the language**, paraphrase, or reinterpret.
- Output the **corrected version of the text only**, with no explanation, no formatting, and no bullet points.

IMPORTANT:
- Remove or normalize all Unicode artifacts.
- Join broken words and fix spacing.
- Fix words where OCR has inserted or swapped incorrect letters or numbers.
- Restore original punctuation and capitalization as best as possible.
- Keep line breaks and paragraphing consistent if found.


OCR text:
{input_text}

Cleaned text:
"""

def pre_clean_text(text):
    # 1. Fix Unicode errors (quotes, dashes, accents)
    text = fix_text(text)

    # 2. Normalize and remove non-ASCII
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # 3. (Optional) Remove garbage characters like long dashes, artifacts
    text = re.sub(r"[^a-zA-Z0-9\s.,;:'\"!?()\[\]\-–—]", "", text)

    # 4. Normalize spacing
    text = re.sub(r"\s+", " ", text).strip()

    return text

def load_api_key(path="key.txt"):
    with open(path, "r") as f:
        return f.read().strip()

API_KEY = load_api_key()

def estimate_tokens(text, chars_per_token=4):
    return len(text) // chars_per_token

def split_text_by_token_estimate(text, max_tokens, chars_per_token=4):
    max_chars = max_tokens * chars_per_token
    chunks = []
    current = ""

    for line in text.splitlines():
        if not line.strip():
            continue
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current.strip())
            current = line
    if current:
        chunks.append(current.strip())

    return chunks


with open("data/eng/the_vampyre_ocr.json", "r") as f_ocr:
    ocr_data = json.load(f_ocr)

# with open("data/eng/the_vampyre_clean.json", "r") as f_clean:
#     clean_data = json.load(f_clean)

max_token_input = 2000
first_key = list(ocr_data.keys())[0]
first_text = ocr_data[first_key]
first_text = pre_clean_text(first_text)     #remove unicode codes
chunks = split_text_by_token_estimate(first_text, max_tokens=max_token_input, chars_per_token=4)

#answers
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
            "temperature": 0.3
        }
    )

    result = response.json()["choices"][0]["message"]["content"].strip()
    cleaned_chunks.append(result)
    print(f"✔️ Chunk {idx + 1}/{len(chunks)} done")

# Combina i pezzi
final_text = "\n".join(cleaned_chunks)

# Salva il risultato
output = {
    "id": first_key,
    "ocr_text": first_text,
    "cleaned_text": final_text
}

output_path = "results/deepseek_cleaned_chunked_v5.json"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Testo completo ripulito salvato in: {output_path}")
