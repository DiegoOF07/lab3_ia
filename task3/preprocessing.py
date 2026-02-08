import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_base_dir():
    return os.path.dirname(os.path.abspath(__file__))

# Se eliminan todas las columnas del equipo rojo y las columnas de diferencias
# (blueGoldDiff, blueExperienceDiff) para evitar data leakage, ya que al minuto
# 10 solo se debería utilizar información observable del equipo azul.

def clean_and_save_dataset():
    base_dir = get_base_dir()
    input_path = os.path.join(base_dir, "data", "dataset.txt")
    output_path = os.path.join(base_dir, "data", "dataset_clean.txt")

    with open(input_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = list(reader)

    keep_indices = []
    for i, col in enumerate(header):
        if col.startswith("red"):
            continue
        if col in ("blueGoldDiff", "blueExperienceDiff"):
            continue
        keep_indices.append(i)

    clean_header = [header[i] for i in keep_indices]
    clean_rows = [[row[i] for i in keep_indices] for row in rows]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(clean_header)
        writer.writerows(clean_rows)

    return clean_header, clean_rows


def load_clean_dataset():
    base_dir = get_base_dir()
    path = os.path.join(base_dir, "data", "dataset_clean.txt")

    with open(path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = list(reader)

    return header, rows


def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # El escalado es necesario para SVM porque este modelo se basa en distancias
    # entre puntos para encontrar el hiperplano óptimo. Sin escalado, variables
    # con mayor magnitud dominarían el modelo. Los Árboles de Decisión no requieren
    # escalado ya que toman decisiones basadas en comparaciones y no en distancias.

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test
