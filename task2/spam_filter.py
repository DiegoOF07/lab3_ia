import math
import pandas as pd
from collections import Counter
from parser import clean_message

class SpamFilter:
    def __init__(self, k=1):
        self.k = k
        self.vocabulary = set()
        self.word_counts = {"spam": Counter(), "ham": Counter()}
        self.class_counts = Counter()
        self.total_words = Counter()
        self.priors = {}
        self.likelihoods = {"spam": {}, "ham": {}}
    
    def train(self, train_data):
        for label, message in train_data:
            self.class_counts[label] += 1
            words = clean_message(message)
            self.word_counts[label].update(words)
            self.total_words[label] += len(words)
        
        self.vocabulary = set(self.word_counts["spam"].keys()) | set(self.word_counts["ham"].keys())
        vocab_size = len(self.vocabulary)
        
        total_messages = sum(self.class_counts.values())
        self.priors["spam"] = self.class_counts["spam"] / total_messages
        self.priors["ham"] = self.class_counts["ham"] / total_messages
        
        for word in self.vocabulary:
            self.likelihoods["spam"][word] = (self.word_counts["spam"][word] + self.k) / (self.total_words["spam"] + self.k * vocab_size)
            self.likelihoods["ham"][word] = (self.word_counts["ham"][word] + self.k) / (self.total_words["ham"] + self.k * vocab_size)
    
    def predict(self, message: str) -> str:
        words = clean_message(message)
        
        log_prob_spam = math.log(self.priors["spam"])
        log_prob_ham = math.log(self.priors["ham"])
        
        for word in words:
            if word in self.vocabulary:
                log_prob_spam += math.log(self.likelihoods["spam"][word])
                log_prob_ham += math.log(self.likelihoods["ham"][word])
        
        return "spam" if log_prob_spam > log_prob_ham else "ham"
    
    def evaluate(self, test_data) -> dict:
        confusion_matrix = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        
        for true_label, message in test_data:
            predicted_label = self.predict(message)
            
            if true_label == "spam" and predicted_label == "spam":
                confusion_matrix["TP"] += 1
            elif true_label == "ham" and predicted_label == "spam":
                confusion_matrix["FP"] += 1
            elif true_label == "ham" and predicted_label == "ham":
                confusion_matrix["TN"] += 1
            elif true_label == "spam" and predicted_label == "ham":
                confusion_matrix["FN"] += 1
        
        total = len(test_data)
        correct = confusion_matrix["TP"] + confusion_matrix["TN"]
        accuracy = correct / total if total > 0 else 0
        
        precision = confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FP"]) if (confusion_matrix["TP"] + confusion_matrix["FP"]) > 0 else 0
        recall = confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FN"]) if (confusion_matrix["TP"] + confusion_matrix["FN"]) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "confusion_matrix": confusion_matrix,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
    
    def get_likelihoods(self):
        data = []
        for word in self.vocabulary:
            spam_prob = self.likelihoods["spam"][word]
            ham_prob = self.likelihoods["ham"][word]
            data.append({
                'word': word,
                'P(word|Spam)': spam_prob,
                'P(word|Ham)': ham_prob,
            })
        
        df = pd.DataFrame(data)
        df = df.set_index('word')
        return df
