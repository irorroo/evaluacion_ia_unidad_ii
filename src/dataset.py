"""
dataset.py  (Pregunta 1)
------------------------
Se encarga de:
  1. Fijar la semilla individual (68) para que todo sea reproducible.
  2. Descargar y cargar CIFAR-10 con torchvision.
  3. Visualizar una cuadrícula de imágenes con su etiqueta real.
  4. Definir el pipeline de transformaciones (ToTensor + Normalize + Data Augmentation).
  5. Dividir el set de entrenamiento en train/validación de forma ESTRATIFICADA.

Semilla: mi código de matrícula es 022101068E -> los dos últimos DÍGITOS son 68.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Configuración global
# ----------------------------------------------------------------------
SEED = 68  # semilla individual (dos últimos dígitos del código 022101068E)

# Rutas relativas a la RAÍZ del proyecto (así funcionan igual desde src/ o notebooks/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

# Las 10 clases de CIFAR-10 (el índice 0..9 del target corresponde a este orden)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

# Media y desviación estándar de CADA canal RGB de CIFAR-10 (calculadas sobre el train set).
# Se usan en Normalize para dejar cada canal con media ~0 y desviación ~1.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ----------------------------------------------------------------------
# 1. Semilla
# ----------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    """
    Fija TODAS las fuentes de aleatoriedad para que los resultados sean reproducibles:
      - random / numpy  : mezcla de listas, índices, etc.
      - torch           : inicialización aleatoria de los pesos de la red, dropout, etc.
      - cuda            : lo mismo, pero en GPU (si existe).
    """
    random.seed(seed)
    np.random.seed(seed)          # lo pide el enunciado
    torch.manual_seed(seed)       # lo pide el enunciado
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Hace que cuDNN use algoritmos deterministas (mismos resultados en cada corrida)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# 2. Transformaciones
# ----------------------------------------------------------------------
def get_transforms():
    """
    Devuelve dos pipelines:

    train_tf (con Data Augmentation) -> SOLO para entrenar:
        - RandomCrop(32, padding=4): rellena 4 px por cada borde y recorta 32x32 al azar
          (la imagen se "desplaza" un poco). Ayuda a que la red no memorice posiciones.
        - RandomHorizontalFlip: voltea la imagen en espejo con prob. 0.5.
        - ToTensor: convierte la imagen PIL (H,W,C) con valores 0-255 a tensor (C,H,W) en [0,1].
        - Normalize: (x - media) / desviación por canal RGB.

    eval_tf (SIN augmentation) -> para validación y test:
        Solo ToTensor + Normalize. Evaluar con imágenes aleatoriamente alteradas daría
        métricas ruidosas, por eso a validación/test NO se les aplica augmentation.
    """
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_tf, eval_tf


# ----------------------------------------------------------------------
# 3. Carga + partición estratificada train/validación
# ----------------------------------------------------------------------
def load_and_split(val_size: float = 0.1, seed: int = SEED):
    """
    Descarga CIFAR-10 (si no está) y devuelve (train_set, val_set, test_set).

    - CIFAR-10 trae 50 000 imágenes de train y 10 000 de test.
    - Partimos las 50 000 en 45 000 (train) y 5 000 (validación) con train_test_split
      usando stratify=targets: cada clase queda con la MISMA proporción en ambos
      subconjuntos (4500 y 500 imágenes por clase). random_state=seed hace que la
      mezcla dependa de mi semilla (68).

    Detalle importante: creamos DOS copias del train set, una con augmentation y otra sin.
    Train usa la copia con augmentation; validación usa la copia SIN augmentation, pero
    ambas con los mismos índices (Subset) para que no se crucen imágenes.
    """
    train_tf, eval_tf = get_transforms()

    full_train_aug = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
    full_train_eval = datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=eval_tf)
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=eval_tf)

    targets = np.array(full_train_aug.targets)   # etiqueta (0-9) de cada imagen
    indices = np.arange(len(targets))            # 0, 1, 2, ..., 49999

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        stratify=targets,      # <- partición ESTRATIFICADA
        shuffle=True,
        random_state=seed,     # <- depende de la semilla individual
    )

    train_set = Subset(full_train_aug, train_idx)
    val_set = Subset(full_train_eval, val_idx)

    # --- Impresión de dimensiones (lo pide el enunciado) ---
    print("=" * 55)
    print("PARTICIÓN DEL DATASET (semilla = %d)" % seed)
    print("=" * 55)
    print(f"Train      : {len(train_set):>6} imágenes -> tensor (N, 3, 32, 32) = ({len(train_set)}, 3, 32, 32)")
    print(f"Validación : {len(val_set):>6} imágenes -> tensor (N, 3, 32, 32) = ({len(val_set)}, 3, 32, 32)")
    print(f"Test       : {len(test_set):>6} imágenes -> tensor (N, 3, 32, 32) = ({len(test_set)}, 3, 32, 32)")
    print("Imágenes por clase en train      :", np.bincount(targets[train_idx]).tolist())
    print("Imágenes por clase en validación :", np.bincount(targets[val_idx]).tolist())

    return train_set, val_set, test_set


def get_dataloaders(batch_size: int = 128, num_workers: int = 0):
    """
    Envuelve los datasets en DataLoaders (entregan lotes/batches de tensores 4D).
    Cada lote de imágenes tiene forma (Batch_Size, Channels, Height, Width) = (128, 3, 32, 32).

    num_workers=0 evita problemas de multiprocesamiento en Windows.
    El Generator con semilla hace que el orden del shuffle sea reproducible.
    """
    train_set, val_set, test_set = load_and_split()

    g = torch.Generator()
    g.manual_seed(SEED)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              generator=g, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader


# ----------------------------------------------------------------------
# 4. Visualización de muestras
# ----------------------------------------------------------------------
def visualize_samples(n: int = 16, save_path: str = None, show: bool = False) -> None:
    """
    Muestra una cuadrícula de n imágenes (4 columnas) con su etiqueta REAL.
    Se usan las imágenes originales (sin normalizar) para que se vean con colores normales.
    Las n imágenes se eligen al azar, pero con la semilla fija, así que siempre son las mismas.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "sample_grid.png")

    raw = datasets.CIFAR10(DATA_DIR, train=True, download=True)  # sin transform -> imágenes PIL
    rng = np.random.RandomState(SEED)
    chosen = rng.choice(len(raw), size=n, replace=False)

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows))
    for ax, idx in zip(np.array(axes).flatten(), chosen):
        img, label = raw[idx]
        ax.imshow(img)
        ax.set_title(CLASSES[label], fontsize=11)   # etiqueta real de la foto
        ax.axis("off")
    plt.suptitle(f"Muestras de CIFAR-10 (semilla {SEED})", fontsize=13)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[dataset] Cuadrícula guardada en: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------------------------------------------------
# Prueba aislada:  python src/dataset.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(SEED)
    visualize_samples(n=16)
    train_loader, val_loader, test_loader = get_dataloaders()

    images, labels = next(iter(train_loader))
    print("\nForma de un batch de imágenes :", tuple(images.shape), "-> (Batch, Canales, Alto, Ancho)")
    print("Forma de un batch de etiquetas:", tuple(labels.shape))
