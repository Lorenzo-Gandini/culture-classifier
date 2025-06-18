from torch.utils.data.dataset import Dataset
import os 
import json 
from sklearn.model_selection import train_test_split
import torch 

class OCRDataset(Dataset):
    def __init__(self, data, tokenizer, max_length = 4096):
        super().__init__()
        self.dataset = data  # Each item should be a (prompt, target) tuple
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        prompt = self.dataset[index][0]
        target = self.dataset[index][1]


        chat = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": target}]
        

        enc = self.tokenizer.apply_chat_template(conversation=chat, 
                                                 tokenize=True,
                                                 add_generation_prompt=True,
                                                 return_tensors="pt",
                                                 padding="max_length",
                                                 max_length=self.max_length,
                                                 return_dict = True)
        
        
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        
        # Find where the assistant response starts
        assistant_token = self.tokenizer.encode("[INST]", add_special_tokens=False)[0]
        # Create labels
        labels = input_ids.clone()
        
            # Tokenize the full special assistant start sequence
        assistant_start_seq = "<|eot_id|><|start_header_id|> assistant<|end_header_id|>"
        assistant_start_ids = self.tokenizer.encode(assistant_start_seq, add_special_tokens=False)
        assistant_start_ids = torch.tensor(assistant_start_ids, device=input_ids.device)

        # Find the start position of this subsequence inside input_ids
        assistant_pos = find_subsequence(input_ids, assistant_start_ids)

        if assistant_pos >= 0:
            # Mask everything up to and including the assistant start token sequence
            labels[:assistant_pos + len(assistant_start_ids)] = -100
            labels[attention_mask == 0] = -100
        else:
            # Fallback: mask everything (should not happen)
            labels[:] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def find_subsequence(sequence, subsequence):
    """
    Find the first occurrence of subsequence in sequence.
    Returns the start index or -1 if not found.
    """
    seq_len = len(sequence)
    sub_len = len(subsequence)
    for i in range(seq_len - sub_len + 1):
        if torch.equal(sequence[i:i+sub_len], subsequence):
            return i
    return -1


def retrieve_datasets(path, test_size = 0.1, random_state=42):
    
    """
        Function that opens and returns the files in dictionary format
    """

    clean_path = os.path.join(path, "clean.json")
    noisy_path = os.path.join(path, "noisy.json")

    try:
        with open(clean_path, "r+") as f:
            clean_file = json.load(f)

        with open(noisy_path, "r+") as f:
            noisy_file = json.load(f)
    except:
        raise Exception("Error opening files")


    keys = sorted(int(k) for k in clean_file.keys())
    paired_data = [(noisy_file[str(k)], clean_file[str(k)]) for k in keys]

    train_data, test_data = train_test_split(paired_data, test_size=test_size, random_state=random_state)
    return train_data, test_data



if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Adjust these paths to your actual data folder
    data_path = "./data_preprocessed/eng"  

    # Load train and test splits
    train_data, test_data = retrieve_datasets(data_path)

    print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")

    # Initialize your tokenizer (change model name to your model/tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("sapienzanlp/Minerva-7B-instruct-v1.0")

    # Create dataset instance for training data
    train_dataset = OCRDataset(train_data, tokenizer)

    # Test retrieving a sample
    sample = train_dataset[3]

    # Print sample keys
    print("Sample keys:", sample.keys())

    # Decode input_ids and labels for inspection
    input_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
    labels = sample["labels"].tolist()
    
    # To decode labels, mask out -100 tokens
    label_tokens = [id if id != -100 else tokenizer.pad_token_id for id in labels]
    label_text = tokenizer.decode(label_tokens, skip_special_tokens=False)

    print("\n--- Sample Input Text ---")
    print(input_text)

    print("\n--- Sample Labels Text ---")
    print(label_text)

    # Check that tokens before <|assistant|> are masked (-100)
    assistant_token_id = tokenizer.encode("<|assistant|>", add_special_tokens=False)[0]
    assistant_pos = sample["input_ids"].tolist().index(assistant_token_id)

    masked_before_assistant = all(l == -100 for l in labels[:assistant_pos + 1])
    print(f"\nTokens before and including <|assistant|> masked? {masked_before_assistant}")