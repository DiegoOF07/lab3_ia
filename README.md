# Laboratorio 3 - Inteligencia Artificial

## Descripción

Este repositorio contiene el Laboratorio 3 del curso de Inteligencia Artificial. El proyecto implementa tres algoritmos de clasificación supervisada (Naive Bayes, Support Vector Machines y Árboles de Decisión) aplicados a dos problemas distintos: detección de spam y predicción de resultados en partidas de e-sports (League of Legends). Se incluyen implementaciones desde cero y comparaciones con modelos de scikit-learn.

## Características

### Task 2: Filtro de Spam Bayesiano

- Implementación manual de clasificador Naive Bayes desde cero
- Preprocesamiento de texto y construcción de "Bag of Words"
- Aplicación de Laplace Smoothing para evitar overfitting
- Predicción con log-probabilities para evitar underflow numérico
- Evaluación con matriz de confusión y accuracy

### Task 3: Clasificación en e-sports (League of Legends)

- Preprocesamiento de datos de partidas (eliminación de redundancias, escalado)
- Entrenamiento de SVM con kernels lineal y RBF
- Entrenamiento y visualización de Árboles de Decisión
- Análisis de importancia de features para interpretabilidad
- Comparación de modelos basada en accuracy e interpretabilidad

## Dependencias

Para ejecutar este proyecto, necesitas instalar las siguientes dependencias:

```bash
pip install numpy pandas matplotlib scikit-learn nltk seaborn
```

## Cómo correrlo

### Task 2
1. Asegúrate de estar en el directorio task2
```bash
cd lab3_ia/task2
```
2. Ejecuta el script principal:

```bash
python main.py
```

### Task 3
1. Asegúrate de estar en el directorio task3
```bash
cd lab3_ia/task3
```
2. Ejecuta el script principal:

```bash
python main.py
```
