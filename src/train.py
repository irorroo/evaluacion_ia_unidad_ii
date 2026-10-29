"""
train.py  (Pregunta 4)
----------------------
Entrena la CNN con CrossEntropyLoss + Adam, registra Loss y Accuracy por época
(train y validación), evalúa en el set de test y guarda en outputs/:
    - loss_accuracy.png     (curvas de aprendizaje)
    - confusion_matrix.png  (matriz de confusión sobre test)

Ejecutar desde la raíz del proyecto:
    python src/train.py
"""

import os
import sys
import time

# Permite hacer "from src.xxx import ..." aunque se ejecute como "python src/train.py"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from src.dataset import SEED, CLASSES, OUTPUT_DIR, set_seed, get_dataloaders
from src.model import CNN, count_parameters

# ----------------------------------------------------------------------
# Hiperparámetros
# ----------------------------------------------------------------------
EPOCHS = 8         # número de pasadas completas por el set de entrenamiento
BATCH_SIZE = 128       # imágenes por lote
LEARNING_RATE = 1e-3   # tasa de aprendizaje de Adam


# ----------------------------------------------------------------------
# Una época de ENTRENAMIENTO
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  # activa Dropout
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()              # 1) limpia los gradientes del paso anterior
        outputs = model(images)            # 2) forward: logits (B, 10)
        loss = criterion(outputs, labels)  # 3) pérdida CrossEntropy
        loss.backward()                    # 4) backpropagation: calcula gradientes
        optimizer.step()                   # 5) Adam actualiza los pesos

        running_loss += loss.item() * images.size(0)             # suma ponderada por tamaño del lote
        correct += (outputs.argmax(dim=1) == labels).sum().item()  # aciertos (clase con mayor logit)
        total += labels.size(0)

    return running_loss / total, correct / total   # loss promedio, accuracy


# ----------------------------------------------------------------------
# Evaluación (validación o test): sin gradientes y sin actualizar pesos
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()  # desactiva Dropout
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return (running_loss / total, correct / total,
            np.concatenate(all_labels), np.concatenate(all_preds))


# ----------------------------------------------------------------------
# Gráficas
# ----------------------------------------------------------------------
def plot_learning_curves(history, save_path):
    """Curvas de Loss y Accuracy (train vs validación) por época."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], marker="o", label="Train")
    ax1.plot(epochs, history["val_loss"], marker="o", label="Validación")
    ax1.set_title("Loss por época")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss (CrossEntropy)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], marker="o", label="Train")
    ax2.plot(epochs, history["val_acc"], marker="o", label="Validación")
    ax2.set_title("Accuracy por época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.suptitle(f"Curvas de aprendizaje (semilla {SEED})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] Curvas guardadas en: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Matriz de confusión: filas = clase real, columnas = clase predicha."""
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Matriz de confusión - Test set")
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] Matriz de confusión guardada en: {save_path}")


# ----------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------
def main():
    set_seed(SEED)  # semilla ANTES de crear el modelo (define los pesos iniciales)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device} | Semilla: {SEED} | Épocas: {EPOCHS}")

    # 1) Datos
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # 2) Modelo, pérdida y optimizador
    model = CNN().to(device)
    print(f"Parámetros entrenables: {count_parameters(model):,}")
    criterion = nn.CrossEntropyLoss()                                 # pérdida para clasificación multiclase
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)      # optimizador Adam

    # 3) Bucle de entrenamiento por épocas
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_path = os.path.join(OUTPUT_DIR, "best_model.pt")

    print("\n" + "=" * 78)
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Guardamos los pesos de la mejor época según validación
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

        print(f"Época {epoch + 1:>2}/{EPOCHS} | "
              f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"Val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"{time.time() - t0:.1f}s")
    print("=" * 78)

    # 4) Evaluación final en TEST con los mejores pesos (según validación)
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    print("\nRESULTADOS FINALES")
    print(f"Mejor Val Accuracy : {best_val_acc:.4f}")
    print(f"Test Loss          : {test_loss:.4f}")
    print(f"Test Accuracy      : {test_acc:.4f}")
    gap = history["train_acc"][-1] - history["val_acc"][-1]
    print(f"Última época -> Train acc {history['train_acc'][-1]:.4f} | "
          f"Val acc {history['val_acc'][-1]:.4f} | Brecha (train - val): {gap:+.4f}")
    print("\nReporte de clasificación (test):")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))

    # 5) Artefactos gráficos obligatorios
    plot_learning_curves(history, os.path.join(OUTPUT_DIR, "loss_accuracy.png"))
    plot_confusion_matrix(y_true, y_pred, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))


if __name__ == "__main__":
    main()
