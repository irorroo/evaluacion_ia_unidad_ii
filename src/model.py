"""
model.py  (Pregunta 3)
----------------------
Define la CNN en PyTorch y valida el flujo de tensores 4D con un forward de prueba.

Formato de los tensores 4D:  (Batch_Size, Channels, Height, Width)

Flujo de shapes (con Batch = B):
    Entrada        : (B,   3, 32, 32)
    Bloque 1       : Conv2d(3->32)   + ReLU + MaxPool2d(2)  -> (B,  32, 16, 16)
    Bloque 2       : Conv2d(32->64)  + ReLU + MaxPool2d(2)  -> (B,  64,  8,  8)
    Bloque 3       : Conv2d(64->128) + ReLU + MaxPool2d(2)  -> (B, 128,  4,  4)
    Flatten        :                                         -> (B, 2048)   [128*4*4]
    Linear + ReLU  : 2048 -> 256                             -> (B, 256)
    Dropout(0.5)   :                                         -> (B, 256)
    Linear (salida): 256 -> 10 clases                        -> (B, 10)
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # ---------------- Bloques convolucionales (extractor de características) ----------------
        # kernel_size=3 y padding=1 -> la convolución NO cambia el alto/ancho (32 sigue en 32).
        # MaxPool2d(2) -> divide alto y ancho entre 2 (32 -> 16 -> 8 -> 4).
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # ---------------- Parte densa (clasificador) ----------------
        # Flatten: (B, 128, 4, 4) -> (B, 2048). Es la "conexión" entre lo convolucional y lo denso.
        # 2048 = 128 canales * 4 alto * 4 ancho. Si este número está mal -> Shape Mismatch.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),          # apaga el 50% de neuronas al entrenar: reduce overfitting
            nn.Linear(256, num_classes),  # 10 salidas (logits), una por clase
        )

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        x: tensor 4D (B, 3, 32, 32).
        Devuelve logits (B, 10). No se aplica Softmax porque CrossEntropyLoss ya lo incluye.
        Con verbose=True imprime el shape después de cada etapa.
        """
        if verbose:
            print(f"Entrada                : {tuple(x.shape)}")

        x = self.block1(x)
        if verbose:
            print(f"Tras bloque conv 1     : {tuple(x.shape)}")

        x = self.block2(x)
        if verbose:
            print(f"Tras bloque conv 2     : {tuple(x.shape)}")

        x = self.block3(x)
        if verbose:
            print(f"Tras bloque conv 3     : {tuple(x.shape)}")

        x = self.classifier[0](x)   # Flatten
        if verbose:
            print(f"Tras Flatten           : {tuple(x.shape)}")

        x = self.classifier[1:](x)  # Linear -> ReLU -> Dropout -> Linear
        if verbose:
            print(f"Salida (logits)        : {tuple(x.shape)}")
        return x


def count_parameters(model: nn.Module) -> int:
    """Cuenta los parámetros entrenables del modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_forward_pass(batch_size: int = 8) -> None:
    """
    Validación de shapes (aserción): pasa un batch SIMULADO de imágenes aleatorias por la red,
    imprime los shapes intermedios y comprueba con assert que la salida es (batch, 10).
    Si las dimensiones de la capa Linear estuvieran mal, aquí saltaría el error (Shape Mismatch).
    """
    model = CNN()
    model.eval()

    dummy_batch = torch.randn(batch_size, 3, 32, 32)  # batch simulado 4D
    print("=" * 50)
    print("VALIDACIÓN DE SHAPES (forward pass de prueba)")
    print("=" * 50)
    with torch.no_grad():
        out = model(dummy_batch, verbose=True)

    assert out.shape == (batch_size, 10), f"Shape inesperado: {tuple(out.shape)}"
    print("-" * 50)
    print(f"ASERCIÓN OK: salida {tuple(out.shape)} == ({batch_size}, 10)")
    print(f"Parámetros entrenables totales: {count_parameters(model):,}")


# Prueba aislada:  python src/model.py
if __name__ == "__main__":
    test_forward_pass()
