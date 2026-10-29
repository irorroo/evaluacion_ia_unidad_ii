# Clasificación de imágenes CIFAR-10 mediante CNN (PyTorch)

**Evaluación de la Segunda Unidad – Inteligencia Artificial**
Universidad Andina del Cusco – Escuela Profesional de Ingeniería de Sistemas

- **Estudiante:** Rodrigo Armando Oviedo Vargas
- **Código de matrícula:** 022101068E
- **Semilla individual:** `68` (dos últimos dígitos del código) aplicada con `torch.manual_seed(68)` y `np.random.seed(68)`

---

## 1. Estructura del repositorio

```
cnn-cifar10/
├── src/
│   ├── dataset.py      # Carga, semilla, transformaciones, partición estratificada y visualización (Pregunta 1)
│   ├── model.py        # Arquitectura CNN y validación de shapes (Pregunta 3)
│   └── train.py        # Bucle de entrenamiento, métricas y gráficas (Pregunta 4)
├── notebooks/
│   └── experiment.ipynb  # EDA, visualización de muestras y pruebas
├── outputs/
│   ├── sample_grid.png       # Cuadrícula de 16 imágenes con su etiqueta real
│   ├── loss_accuracy.png     # Curvas de aprendizaje
│   └── confusion_matrix.png  # Matriz de confusión sobre el set de test
├── requirements.txt
└── README.md
```

## 2. Cómo ejecutar

```bash
pip install -r requirements.txt
python src/dataset.py     # descarga CIFAR-10, guarda la cuadrícula y muestra la partición
python src/model.py       # valida los shapes del forward pass
python src/train.py       # entrena, evalúa y guarda las gráficas en outputs/
```

## 3. Pipeline de datos (Pregunta 1)

| Aspecto | Decisión |
|---|---|
| Dataset | CIFAR-10 (60 000 imágenes RGB de 32×32, 10 clases) vía `torchvision.datasets.CIFAR10` |
| Partición | 50 000 de train → **45 000 train / 5 000 validación** (estratificada con `train_test_split(stratify=...)`, `random_state=68`); 10 000 de test |
| Normalización | `Normalize` con media y desviación estándar por canal RGB de CIFAR-10 |
| Data Augmentation (solo train) | `RandomCrop(32, padding=4)` y `RandomHorizontalFlip(p=0.5)` |
| Validación / Test | Sin augmentation, únicamente `ToTensor` + `Normalize` |

## 4. Justificación teórica: CNN frente a MLP (Pregunta 2)

### 4.1 Limitaciones de un Perceptrón Multicapa (MLP) con imágenes

Un MLP exige que la entrada sea un vector unidimensional. Una imagen de CIFAR-10 tiene forma 3×32×32, por lo que debe aplanarse en un vector de 3 072 valores, lo cual genera tres problemas:

1. **Explosión de parámetros.** Una sola capa oculta de 1 024 neuronas requiere 3 072 × 1 024 + 1 024 ≈ **3,1 millones de parámetros**, cinco veces más que toda la CNN propuesta (≈ 0,62 millones). Con imágenes de mayor resolución el crecimiento es inviable, con el consecuente riesgo de sobreajuste y alto costo computacional.
2. **Pérdida de la estructura espacial.** Al aplanar, píxeles vecinos y píxeles lejanos se tratan igual; el MLP no sabe qué píxeles están cerca entre sí y debe aprender esa relación desde cero.
3. **Ausencia de invarianza a la posición.** Si un objeto se desplaza unos píxeles, el vector de entrada cambia por completo y las neuronas que detectaban el patrón en otra posición ya no responden. El MLP tendría que aprender el mismo patrón por separado en cada posición posible.

### 4.2 Propiedades de las CNN que resuelven estas limitaciones

- **Conexiones locales (campo receptivo local):** cada neurona de una capa convolucional solo observa una región pequeña (por ejemplo 3×3 píxeles), aprovechando que en las imágenes la información relevante (bordes, texturas) es local.
- **Weight Sharing (pesos compartidos):** el mismo filtro (kernel) se desliza sobre toda la imagen, reutilizando sus pesos en cada posición. Por ejemplo, `Conv2d(3→32, kernel 3×3)` tiene solo 3×3×3×32 + 32 = **896 parámetros**, independientemente del tamaño de la imagen.
- **Invarianza/equivarianza espacial:** como el mismo filtro se aplica en todas las posiciones, un patrón se detecta sin importar dónde aparezca; además, `MaxPool2d` aporta tolerancia a pequeños desplazamientos y reduce la resolución progresivamente.
- **Jerarquía de características:** las primeras capas detectan bordes y colores, las intermedias texturas y formas, y las últimas partes de objetos, que las capas densas finales combinan para clasificar.

## 5. Arquitectura de la CNN (Pregunta 2.2 y 3)

Tres bloques convolucionales (`Conv2d` → `ReLU` → `MaxPool2d`) seguidos de un clasificador denso. Formato de tensores: `(Batch, Channels, Height, Width)`.

| # | Capa | Configuración | Shape de salida | Parámetros |
|---|---|---|---|---|
| – | Entrada | imagen normalizada | (B, 3, 32, 32) | – |
| 1 | Conv2d + ReLU | 3 → 32, kernel 3×3, padding 1 | (B, 32, 32, 32) | 896 |
| 1 | MaxPool2d | 2×2 | (B, 32, 16, 16) | 0 |
| 2 | Conv2d + ReLU | 32 → 64, kernel 3×3, padding 1 | (B, 64, 16, 16) | 18 496 |
| 2 | MaxPool2d | 2×2 | (B, 64, 8, 8) | 0 |
| 3 | Conv2d + ReLU | 64 → 128, kernel 3×3, padding 1 | (B, 128, 8, 8) | 73 856 |
| 3 | MaxPool2d | 2×2 | (B, 128, 4, 4) | 0 |
| – | Flatten | 128 × 4 × 4 | (B, 2048) | 0 |
| 4 | Linear + ReLU | 2048 → 256 | (B, 256) | 524 544 |
| – | Dropout | p = 0,5 | (B, 256) | 0 |
| 5 | Linear (salida) | 256 → 10 | (B, 10) | 2 570 |
| | | **Total** | | **620 362** |

**Hiperparámetros:** función de pérdida `CrossEntropyLoss`, optimizador `Adam` (lr = 1e-3), batch size = 128, épocas = 15, semilla = 68.

## 6. Resultados (Pregunta 4)

> **COMPLETAR** con los valores reales impresos en consola al ejecutar `python src/train.py`.

| Métrica | Valor |
|---|---|
| Train accuracy (última época) | `[COMPLETAR]` |
| Validación accuracy (última época) | `[COMPLETAR]` |
| Mejor validación accuracy | `[COMPLETAR]` |
| Test accuracy | `[COMPLETAR]` |
| Test loss | `[COMPLETAR]` |
| Brecha train − validación | `[COMPLETAR]` |

**Curvas de aprendizaje:**

![Curvas de aprendizaje](outputs/loss_accuracy.png)

**Matriz de confusión (test):**

![Matriz de confusión](outputs/confusion_matrix.png)

**Muestras del dataset:**

![Muestras](outputs/sample_grid.png)

## 7. Análisis crítico (Pregunta 4.4)

> **COMPLETAR** con tus números reales. Guía de interpretación:
>
> - **Overfitting:** el accuracy de train sigue subiendo mientras el de validación se estanca o baja, y el loss de validación empieza a subir. Una brecha train − validación grande (por ejemplo, más de 5–10 puntos porcentuales) es síntoma de sobreajuste.
> - **Underfitting:** train y validación tienen accuracy bajo y parecido, y el loss sigue alto o aún bajando al terminar. Indica que el modelo no tiene capacidad suficiente o necesita más épocas.
> - **Buen ajuste:** ambos accuracy son altos y cercanos, y el test accuracy es similar al de validación.
>
> Redacta: (1) los valores numéricos de brecha y test accuracy, (2) tu conclusión (overfitting / underfitting / ajuste razonable) justificada con esas cifras, (3) qué clases se confunden más según la matriz (en CIFAR-10 suelen ser *cat* ↔ *dog*) y (4) qué mejorarías (más épocas, BatchNorm, más augmentation, scheduler del learning rate).
