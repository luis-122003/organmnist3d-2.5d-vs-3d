"""Figuras 4 y 5 del artículo, a partir de la versión 64³ de OrganMNIST3D (mismos volúmenes que la de 28³).

Uso: python figuras_articulo.py <organmnist3d_64.npz> <carpeta_salida>

Las predicciones son las de la semilla 0: a 64³, las de la corrida OrganMNIST3D_64_Kaggle (versión 2,
fig_predicciones.png en resultados_kaggle_64/, que acierta los 11 volúmenes); a 28³, las de la corrida
OrganMNIST3D_Kaggle (versión 2), tal como salen en fig_predicciones_3d.png y fig_predicciones_2_5d.png.
Los tres scripts sortean los mismos índices de prueba (default_rng(0), una muestra por clase).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASES_ES = ["hígado", "riñón der.", "riñón izq.", "fémur der.", "fémur izq.", "vejiga",
             "corazón", "pulmón der.", "pulmón izq.", "bazo", "páncreas"]
ERRORES = {"3D 64³": {}, "3D 28³": {4: 3}, "2.5D 28³": {9: 0}}   # 3D 28³: fémur izq. -> der.; 2.5D: bazo -> hígado
VERDE, ROJO = "#1a7f37", "#cf222e"

d = np.load(sys.argv[1])
salida = sys.argv[2]
X = {s: d[f"{s}_images"] for s in ("train", "test")}
Y = {s: d[f"{s}_labels"].reshape(-1) for s in ("train", "test")}


def mostrar(ax, corte, v):
    lo, hi = np.percentile(v, [1, 99])
    ax.imshow(corte, cmap="gray", vmin=lo, vmax=hi)
    ax.set_xticks([]); ax.set_yticks([])
    for borde in ax.spines.values():
        borde.set_visible(False)


# Figura 4: el primer volumen de entrenamiento de cada clase, tres cortes centrales.
fig, ax = plt.subplots(3, 11, figsize=(6.0, 1.95), gridspec_kw={"wspace": 0.04, "hspace": 0.04})
for c in range(11):
    v = X["train"][np.where(Y["train"] == c)[0][0]]
    m = v.shape[0] // 2
    for i, corte in enumerate((v[m, :, :], v[:, m, :], v[:, :, m])):
        mostrar(ax[i, c], corte, v)
    ax[0, c].set_title(CLASES_ES[c].replace(" ", "\n"), fontsize=6.3, pad=2)
for i in range(3):
    ax[i, 0].set_ylabel(f"eje {i + 1}", fontsize=6.3, labelpad=2)
fig.savefig(os.path.join(salida, "figura4_muestra_64.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# Figura 5: un volumen de prueba por clase (mismo sorteo que el script) con la predicción de cada red.
rng = np.random.default_rng(0)
idx = [int(rng.choice(np.where(Y["test"] == c)[0], 1)[0]) for c in range(11)]
fig, ax = plt.subplots(2, 6, figsize=(6.0, 3.0), gridspec_kw={"wspace": 0.22, "hspace": 0.52})
for c, (a, i) in enumerate(zip(ax.ravel(), idx)):
    v = X["test"][i]
    mostrar(a, v[v.shape[0] // 2, :, :], v)
    a.set_title(CLASES_ES[c], fontsize=7, fontweight="bold", pad=25)
    for k, (modelo, errores) in enumerate(ERRORES.items()):
        p = errores.get(c, c)
        a.text(0.5, 1.02 + 0.105 * (2 - k), f"{modelo}: {CLASES_ES[p]}", transform=a.transAxes,
               ha="center", va="bottom", fontsize=5.1, color=VERDE if p == c else ROJO)
ax.ravel()[-1].axis("off")
fig.savefig(os.path.join(salida, "figura5_predicciones_64.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("volúmenes de prueba:", idx)
