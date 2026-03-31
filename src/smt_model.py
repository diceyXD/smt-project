import collections
import re
from nltk.translate import ibm1, AlignedSent

class StatisticalMT:
    def __init__(self, direction="en2de"):
        """
        direction: 'en2de' (English to German) or 'de2en' (German to English)
        """
        self.direction = direction
        self.translation_table = None
        self.best_trans = {}
        self.phrase_memory = {}
    
    def train(self, sents, iterations=5):
        print(f"Preparing training data for direction: {self.direction}...")
        training_sents = []
        for s in sents:
            de_words = [w.lower() for w in s.words]
            en_words = [w.lower() for w in s.mots]
            
            # Simple phrase memorization
            if self.direction == "en2de":
                self.phrase_memory[" ".join(en_words)] = " ".join(de_words)
                # IBMModel1 expects AlignedSent(target_words, source_words)
                training_sents.append(AlignedSent(de_words, en_words))
            else:
                self.phrase_memory[" ".join(de_words)] = " ".join(en_words)
                training_sents.append(AlignedSent(en_words, de_words))
                
        print("Training IBM Model 1 (This may take a minute)...")
        self.ibm_model = ibm1.IBMModel1(training_sents, iterations)
        self.translation_table = self.ibm_model.translation_table
        
        print("Building greedy decode dictionary...")
        self.best_trans = {}
        for tgt_word, src_dict in self.translation_table.items():
            for src_word, prob in src_dict.items():
                if src_word is None: continue # NULL token
                if src_word not in self.best_trans or prob > self.best_trans[src_word][1]:
                    self.best_trans[src_word] = (tgt_word, prob)
                    
    def translate(self, text):
        text_lower = text.lower().strip()
        # Direct phrase match fallback (pseudo phrase-based approach)
        if text_lower in self.phrase_memory:
            return self.phrase_memory[text_lower].capitalize()
            
        # Tokenize by keeping punctuation separate
        tokens = re.findall(r"[\w']+|[.,!?;]", text_lower)
        
        translated_tokens = []
        for src_word in tokens:
            if src_word in self.best_trans:
                translated_tokens.append(self.best_trans[src_word][0])
            else:
                translated_tokens.append(src_word) # Keep unknown words directly
                
        # Basic detokenization
        out = " ".join(translated_tokens)
        out = re.sub(r'\s+([.,!?;])', r'\1', out)
        # Capitalize first letter
        if len(out) > 0:
            out = out[0].upper() + out[1:]
        return out
