from torch.utils.data.dataset import Dataset
import os 
import json 
from sklearn.model_selection import train_test_split


   
class OCRDataset(Dataset):

    def __init__(self, data, tokenizer, max_length = 512):
        
        super().__init__()

        self.dataset = data
        self.tokenizer = tokenizer 

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # Get raw input and target text
        input_text = self.dataset[index][0]
        target_text = self.dataset[index][1]

        # Tokenize input and target separately (if causal LM, target is input shifted)
        input_enc = self.tokenizer(input_text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        target_enc = self.tokenizer(target_text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

        # Since tokenizers return tensors with batch dimension, remove it by indexing at 0
        input_ids = input_enc["input_ids"].squeeze(0)
        attention_mask = input_enc["attention_mask"].squeeze(0)
        labels = target_enc["input_ids"].squeeze(0)

        # Replace padding token id with -100 for labels to ignore in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

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