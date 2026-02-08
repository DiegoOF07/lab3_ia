import random

def train_test_split(data, split: float = 0.8, seed: int = 42) -> tuple[list, list]:
    random.seed(seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    split_index = int(len(shuffled_data) * split)
    return shuffled_data[:split_index], shuffled_data[split_index:]

def print_confusion_matrix(cm: dict):
    print("\nMatriz de Confusión:")
    print(f"{'':20} {'Predicho SPAM':>18} {'Predicho HAM':>18}")
    print(f"{'Real SPAM':20} {cm['TP']:>18} {cm['FN']:>18}")
    print(f"{'Real HAM':20} {cm['FP']:>18} {cm['TN']:>18}")

def print_results(results: dict):
    print_confusion_matrix(results["confusion_matrix"])
    
    print(f"\nMétricas de Evaluación:")
    print(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
