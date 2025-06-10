import json
import difflib
import pandas as pd
from collections import Counter
import re
from typing import Dict, List, Tuple, Any

class TestComparator:
    def __init__(self, files: Dict[str, str]):
        self.files = files
        self.texts = {}
        self.data = {}
        
    def load_data(self):
        """Carica i dati dai file JSON"""
        for name, path in self.files.items():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.data[name] = data
                    if "cleaned_text" in data:
                        self.texts[name] = data["cleaned_text"]
                    elif "text" in data:
                        self.texts[name] = data["text"]
                    else:
                        self.texts[name] = self._extract_text_from_data(data)
            except Exception as e:
                print(f"✗ Errore caricando {name}: {e}")
                self.texts[name] = ""
    
    def _extract_text_from_data(self, data: Any) -> str:
        """Estrae tutto il testo da una struttura dati complessa"""
        text_parts = []
        
        def extract_strings(obj):
            if isinstance(obj, str):
                text_parts.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_strings(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_strings(item)
        
        extract_strings(data)
        return " ".join(text_parts)
    
    def _get_all_pairs(self):
        """Genera tutte le possibili coppie di modelli"""
        names = list(self.texts.keys())
        return [(names[i], names[j]) for i in range(len(names)) for j in range(i+1, len(names))]
    
    def basic_similarity(self) -> pd.DataFrame:
        """Calcola similarità base con difflib per tutte le coppie"""
        comparisons = []
        
        for name1, name2 in self._get_all_pairs():
            text1, text2 = self.texts[name1], self.texts[name2]
            
            ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
            matcher = difflib.SequenceMatcher(None, text1, text2)
            longest_match = matcher.find_longest_match(0, len(text1), 0, len(text2))
            
            comparisons.append({
                "Modello A": name1,
                "Modello B": name2,
                "Similarità %": round(ratio * 100, 2),
                "Blocco comune più lungo": longest_match.size,
                "Lunghezza A": len(text1),
                "Lunghezza B": len(text2)
            })
        
        return pd.DataFrame(comparisons)
    
    def word_level_analysis(self) -> pd.DataFrame:
        """Analisi a livello di parole per tutte le coppie"""
        results = []
        word_sets = {name: set(re.findall(r'\b\w+\b', text.lower())) 
                    for name, text in self.texts.items()}
        
        for name1, name2 in self._get_all_pairs():
            set1, set2 = word_sets[name1], word_sets[name2]
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            jaccard = (intersection / union) * 100 if union > 0 else 0
            overlap = (intersection / min(len(set1), len(set2))) * 100 if min(len(set1), len(set2)) > 0 else 0
            
            results.append({
                "Modello A": name1,
                "Modello B": name2,
                "Parole uniche A": len(set1),
                "Parole uniche B": len(set2),
                "Parole comuni": intersection,
                "Jaccard Similarity %": round(jaccard, 2),
                "Overlap Coefficient %": round(overlap, 2)
            })
        
        return pd.DataFrame(results)
    
    def structural_comparison(self) -> pd.DataFrame:
        """Confronta la struttura dei JSON per tutte le coppie"""
        results = []
        
        def get_structure(obj, path=""):
            structure = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    structure.append((new_path, type(value).__name__))
                    structure.extend(get_structure(value, new_path))
            elif isinstance(obj, list) and obj:
                new_path = f"{path}[0]"
                structure.append((new_path, type(obj[0]).__name__))
                structure.extend(get_structure(obj[0], new_path))
            return structure
        
        structures = {name: set(get_structure(data)) for name, data in self.data.items()}
        
        for name1, name2 in self._get_all_pairs():
            struct1, struct2 = structures[name1], structures[name2]
            common = len(struct1 & struct2)
            total = len(struct1 | struct2)
            similarity = (common / total) * 100 if total > 0 else 0
            
            results.append({
                "Modello A": name1,
                "Modello B": name2,
                "Campi A": len(struct1),
                "Campi B": len(struct2),
                "Campi comuni": common,
                "Similarità strutturale %": round(similarity, 2)
            })
        
        return pd.DataFrame(results)
    
    def summary_stats(self) -> pd.DataFrame:
        """Statistiche riassuntive per ogni modello"""
        stats = []
        
        for name, text in self.texts.items():
            words = re.findall(r'\b\w+\b', text.lower())
            sentences = re.split(r'[.!?]+', text)
            
            stats.append({
                "Modello": name,
                "Caratteri totali": len(text),
                "Parole totali": len(words),
                "Parole uniche": len(set(words)),
                "Frasi": len([s for s in sentences if s.strip()]),
                "Parole più comuni": ", ".join([word for word, count in Counter(words).most_common(5)])
            })
        
        return pd.DataFrame(stats)
    
    def get_aggregated_stats(self) -> Dict[str, pd.DataFrame]:
        """Calcola statistiche aggregate per ogni modello rispetto a tutti gli altri"""
        # Calcola tutte le similarità
        similarity_df = self.basic_similarity()
        word_df = self.word_level_analysis()
        struct_df = self.structural_comparison()
        
        # Unisci i dataframe
        merged = similarity_df.merge(word_df, on=["Modello A", "Modello B"])
        merged = merged.merge(struct_df, on=["Modello A", "Modello B"])
        
        # Calcola medie per ogni modello
        agg_stats = []
        models = list(self.texts.keys())
        
        for model in models:
            # Filtra dove il modello è A o B
            mask = (merged["Modello A"] == model) | (merged["Modello B"] == model)
            model_df = merged[mask].copy()
            
            # Rinomina le colonne per avere sempre "Current Model" vs "Other Model"
            model_df["Other Model"] = model_df.apply(
                lambda x: x["Modello B"] if x["Modello A"] == model else x["Modello A"], 
                axis=1
            )
            
            # Calcola medie
            agg_stats.append({
                "Modello": model,
                "Similarità media %": round(model_df["Similarità %"].mean(), 2),
                "Jaccard medio %": round(model_df["Jaccard Similarity %"].mean(), 2),
                "Overlap medio %": round(model_df["Overlap Coefficient %"].mean(), 2),
                "Similarità strutturale media %": round(model_df["Similarità strutturale %"].mean(), 2),
                "Numero confronti": len(model_df)
            })
        
        return pd.DataFrame(agg_stats)
    
    def get_all_statistics(self) -> Dict[str, pd.DataFrame]:
        """Esegue tutte le analisi e restituisce i risultati"""
        return {
            "summary_stats": self.summary_stats(),
            "pairwise_comparison": self.basic_similarity(),
            "word_level_analysis": self.word_level_analysis(),
            "structural_comparison": self.structural_comparison(),
            "aggregated_stats": self.get_aggregated_stats()
        }

if __name__ == "__main__":
    files = {
        "llama_v2": "results/meta-llama/meta-llama_cleaned_optimized_v2.json",
        "opengvlab": "results/opengvlab/opengvlab_cleaned_optimized_v2.json",
        "google": "results/google/google_cleaned_optimized_v2.json",
        "nousresearch": "results/nousresearch/nousresearch_cleaned_optimized_v2.json",
        "mistral-nemo": "results/mistral-nemo/mistral-nemo_cleaned_optimized_v2.json",
        "qwen-2.5-7b": "results/qwen-2.5-7b-instruct/qwen-2.5-7b-instruct_cleaned_optimized_v2.json",
        "qwen-2.5-72b": "results/qwen-2.5-72b-instruct/qwen-2.5-72b-instruct_cleaned_optimized_v2.json",
        "qwen-2.5-coder-32b": "results/qwen-2.5-coder-32b-instruct/qwen-2.5-coder-32b-instruct_cleaned_optimized_v2.json",
        "dolphin3.0": "results/dolphin3.0-mistral-24b/dolphin3.0-mistral-24b_cleaned_optimized_v2.json"
    }
    
    comparator = TestComparator(files)
    comparator.load_data()
    
    stats = comparator.get_all_statistics()
    
    # Stampa i risultati in modo ordinato
    for stat_name, df in stats.items():
        print(f"\n=== {stat_name.upper().replace('_', ' ')} ===")
        print(df.to_string(index=False))