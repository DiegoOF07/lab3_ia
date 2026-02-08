from preprocessing import (
    clean_and_save_dataset,
    load_clean_dataset,
    split_and_scale
)
from svm_models import train_svm_linear, train_svm_rbf
from tree_model import train_decision_tree, visualize_tree, get_top_features
from evaluation import print_results

def main():
    # 1. Limpieza
    header, _ = clean_and_save_dataset()

    # 2. Cargar dataset limpio
    header, rows = load_clean_dataset()

    X = [list(map(float, row[2:])) for row in rows]
    y = [int(row[1]) for row in rows]

    # 3. Split + escalado
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale(X, y)

    # 4. SVM
    acc_linear = train_svm_linear(X_train_scaled, X_test_scaled, y_train, y_test)
    acc_rbf = train_svm_rbf(X_train_scaled, X_test_scaled, y_train, y_test)

    # 5. Árbol
    tree_model, acc_tree = train_decision_tree(X_train, X_test, y_train, y_test)
    visualize_tree(tree_model, header[2:])
    top_features = get_top_features(tree_model, header[2:])

    print("\nTop 5 Features Árbol:")
    for feat, val in top_features:
        print(f"{feat}: {val:.4f}")

    # 6. Comparación final
    print_results(acc_linear, acc_rbf, acc_tree)


if __name__ == "__main__":
    main()