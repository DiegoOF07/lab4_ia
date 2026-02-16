# Laboratorio 4 - Inteligencia Artificial

Este repositorio contiene la implementación y comparación de algoritmos fundamentales de optimización y aprendizaje supervisado: **Descenso de Gradiente** (en sus tres variantes) y el modelo de **Perceptrón** para clasificación binaria.

## Descripción

El proyecto se divide en dos experimentos principales:

1.  **Optimización y Regresión:** Se analiza el comportamiento de las variantes de Descenso de Gradiente para ajustar una función polinomial de tercer grado ($2x^3 - 3x^2 + 5x + 3$) con ruido gaussiano. Se mide la relación entre el tiempo de ejecución y la convergencia del error (MSE).
2.  **Clasificación Binaria:** Implementación desde cero de un clasificador Perceptrón utilizando el dataset *Iris*. El modelo busca encontrar la frontera de decisión óptima para separar dos clases de flores basándose en el largo y ancho del sépalo.



## Características

* **Variantes de Descenso de Gradiente:**
    * **Batch GD:** Estable y preciso, utiliza todo el dataset para cada actualización.
    * **Stochastic GD (SGD):** Rápido y ruidoso, actualiza los pesos muestra por muestra.
    * **Mini-batch GD:** El balance ideal, procesa subconjuntos (batches) de datos.
* **Regresión Polinomial:** Ajuste de curvas mediante expansión de características de grado 3.
* **Perceptrón de Rosenblatt:** Modelo lineal de clasificación con función de activación escalón.
* **Visualización Avanzada:** Gráficas de convergencia en escala logarítmica y visualización de fronteras de decisión en 2D.

## Dependencias

El proyecto requiere **Python 3.8+** y las siguientes librerías:

* `numpy`: Para operaciones matriciales.
* `matplotlib`: Para la generación de gráficos.
* `scikit-learn`: Utilizado exclusivamente para cargar el dataset *Iris*.

Puedes instalarlas con el siguiente comando:

```bash
pip install numpy matplotlib scikit-learn
```

## Estructura del Proyecto

* **`main.py`**: Script principal que coordina los experimentos, integra los módulos y genera las visualizaciones finales.
* **`gradient_descent.py`**: Contiene la implementación de las funciones de optimización:
    * *Batch Gradient Descent*
    * *Stochastic Gradient Descent (SGD)*
    * *Mini-batch Gradient Descent*
* **`perceptron.py`**: Define la clase del modelo **Perceptrón** y las utilidades para la carga y preprocesamiento del dataset *Iris*.

---

## Cómo correrlo

Para ejecutar este proyecto en tu entorno local, sigue estos pasos:

1.  **Clonar el repositorio** o descargar los archivos fuente en una carpeta.
2.  **Instalar las dependencias** necesarias (asegúrate de tener `numpy`, `matplotlib` y `scikit-learn` instalados):
    ```bash
    pip install numpy matplotlib scikit-learn
    ```
3.  **Ejecutar el script principal** desde la terminal:
    ```bash
    python main.py
    ```

---

## Resultados Esperados

Al ejecutar el código, el sistema generará los siguientes resultados:

1.  **Gráfica de Optimización**: Una comparativa visual de **"Estabilidad vs Velocidad"**. Podrás observar cómo el *SGD* converge de forma más errática pero rápida en comparación con la estabilidad del *Batch GD*.

2.  **Métrica de Precisión**: Se imprimirá en la terminal el **Accuracy** (exactitud) alcanzado por el Perceptrón tras el entrenamiento.
3.  **Gráfica de Clasificación**: Un *scatter plot* basado en el dataset *Iris* que muestra la **línea de la frontera de decisión** lograda por el Perceptrón, separando claramente las clases de flores.
