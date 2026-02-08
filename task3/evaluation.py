def print_results(acc_linear, acc_rbf, acc_tree):
    print("\nComparación final (Accuracy): ")
    print(f"SVM Lineal: {acc_linear:.4f}")
    print(f"SVM RBF:    {acc_rbf:.4f}")
    print(f"Árbol:     {acc_tree:.4f}")
