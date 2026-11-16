"""OrganMNIST3D a 64x64x64: clasificación multiclase de 11 órganos con una ResNet 3D.

Uso:
    pip install medmnist torch numpy scikit-learn matplotlib pandas
    python organmnist3d_64.py --npz ruta/organmnist3d_64.npz          # 100 épocas, semillas 0 1 2
    python organmnist3d_64.py --npz ruta/organmnist3d_64.npz --epocas 1 --semillas 0   # prueba rápida

Genera: resultados.csv, historial_seed*.csv, metricas_por_clase_seed*.csv, matrices_confusion.npz,
        modelo_seed*.pt, fig_distribucion_clases.png, fig_muestra_cortes.png, fig_curvas.png,
        fig_confusion.png, fig_predicciones.png
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

CLASES = ["liver", "kidney-right", "kidney-left", "femur-right", "femur-left", "bladder",
          "heart", "lung-right", "lung-left", "spleen", "pancreas"]
N_CLASES = 11


# ------------------------------------------------------------------ carga y resumen
def cargar_npz(ruta):
    """Lee el .npz de MedMNIST. Devuelve dict train/val/test -> (X uint8 [N,S,S,S], y int64 [N])."""
    d = np.load(ruta)
    out = {}
    for s in ("train", "val", "test"):
        X, y = d[f"{s}_images"], d[f"{s}_labels"].reshape(-1).astype(np.int64)
        assert X.ndim == 4 and X.shape[1] == X.shape[2] == X.shape[3], X.shape
        out[s] = (X, y)
    tam = {X.shape[1] for X, _ in out.values()}
    assert len(tam) == 1, tam
    return out


def tamano(datos):
    return datos["train"][0].shape[1]


def resumen(datos):
    for s, (X, y) in datos.items():
        cnt = np.bincount(y, minlength=N_CLASES)
        print(f"{s:5s} X={X.shape} dtype={X.dtype} min={X.min()} max={X.max()} | por clase: {cnt.tolist()}")


# ------------------------------------------------------------------ preprocesamiento
def estadisticas_train(X):
    """Media y desviación calculadas SOLO con entrenamiento (evita fuga de datos)."""
    x = X.astype(np.float32) / 255.0
    return float(x.mean()), float(x.std() + 1e-8)


class Organ3D(Dataset):
    """Cada volumen sale como tensor (1, S, S, S)."""

    def __init__(self, X, y, media, desv, aumento=False, seed=0):
        self.X, self.y, self.m, self.s, self.aug = X, y, media, desv, aumento
        self.rng = np.random.default_rng(seed)
        self.desp = max(1, round(2 * X.shape[1] / 28))   # 2 vóxeles a 28³ -> 5 vóxeles a 64³ (mismo desplazamiento relativo)

    def __len__(self):
        return len(self.y)

    def _aumentar(self, v):
        # Sin volteos: izquierda/derecha cambian la clase (kidney-left vs kidney-right, etc.).
        sh = self.rng.integers(-self.desp, self.desp + 1, size=3)            # traslación
        v = np.roll(v, shift=tuple(sh), axis=(0, 1, 2))
        v = v * self.rng.uniform(0.9, 1.1) + self.rng.uniform(-0.05, 0.05)    # contraste y brillo
        v = v + self.rng.normal(0, 0.01, v.shape).astype(np.float32)         # ruido gaussiano suave
        return v

    def __getitem__(self, i):
        v = self.X[i].astype(np.float32) / 255.0
        if self.aug:
            v = self._aumentar(v)
        v = (v - self.m) / self.s
        return torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)).unsqueeze(0), int(self.y[i])


def pesos_clase(y):
    """Pesos inversamente proporcionales a la frecuencia (pérdida ponderada)."""
    cnt = np.bincount(y, minlength=N_CLASES).astype(np.float32)
    return torch.tensor(cnt.sum() / (N_CLASES * np.maximum(cnt, 1)))


def loaders(datos, batch=32, seed=0):
    m, s = estadisticas_train(datos["train"][0])
    ds = {k: Organ3D(*datos[k], m, s, aumento=(k == "train"), seed=seed) for k in datos}
    g = torch.Generator().manual_seed(seed)
    return {k: DataLoader(ds[k], batch_size=batch, shuffle=(k == "train"), generator=g if k == "train" else None)
            for k in ds}


# ------------------------------------------------------------------ modelo
class Bloque(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm3d(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm3d(cout)
        self.sc = None if (stride == 1 and cin == cout) else nn.Sequential(
            nn.Conv3d(cin, cout, 1, stride, bias=False), nn.BatchNorm3d(cout))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + (x if self.sc is None else self.sc(x)))


class ResNet3D(nn.Module):
    """Convolución inicial 3x3x3, 4 etapas de 2 bloques residuales (paso 2 al inicio de las etapas 2-4),
    promedio global y capa lineal de 11 salidas."""

    def __init__(self, anchos=(16, 32, 64, 128), n_clases=N_CLASES):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, anchos[0], 3, 1, 1, bias=False), nn.BatchNorm3d(anchos[0]), nn.ReLU(inplace=True))
        capas, c = [], anchos[0]
        for i, a in enumerate(anchos):
            capas += [Bloque(c, a, 1 if i == 0 else 2), Bloque(a, a, 1)]
            c = a
        self.capas = nn.Sequential(*capas)
        self.fc = nn.Linear(c, n_clases)

    def forward(self, x):
        return self.fc(self.capas(self.stem(x)).mean(dim=(2, 3, 4)))


def n_param(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ------------------------------------------------------------------ entrenamiento y métricas
@torch.no_grad()
def predecir(m, dl, dev):
    m.eval(); P, Y = [], []
    for x, y in dl:
        with torch.autocast(device_type=dev, enabled=(dev == "cuda")):
            P.append(F.softmax(m(x.to(dev)).float(), 1).cpu())
        Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def metricas(P, Y):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
    pred = P.argmax(1)
    try:
        auc = roc_auc_score(Y, P, multi_class="ovr", average="macro", labels=list(range(N_CLASES)))
    except ValueError:
        auc = float("nan")
    nll = float(-np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, None)).mean())   # entropía cruzada media
    return {"acc": accuracy_score(Y, pred), "f1_macro": f1_score(Y, pred, average="macro"),
            "auc_ovr": auc, "nll": nll, "cm": confusion_matrix(Y, pred, labels=list(range(N_CLASES)))}


def entrenar(datos, epocas=100, lr=1e-3, batch=32, seed=0, dev=None):
    """Adam + coseno, entropía cruzada ponderada, precisión mixta en GPU.
    Se queda con la época de mejor F1 macro en validación (en empate, la de menor entropía cruzada
    en validación: con 161 volúmenes la F1 se satura en 1,0 y sola no distingue épocas) y evalúa una
    sola vez en prueba. La memoria pico se mide desde la 2.ª época, para no contar el espacio de
    trabajo que cuDNN usa al elegir algoritmos (benchmark) durante la 1.ª."""
    torch.manual_seed(seed); np.random.seed(seed)
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    dl = loaders(datos, batch, seed)
    m = ResNet3D().to(dev)
    w = pesos_clase(datos["train"][1]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epocas)
    scaler = torch.amp.GradScaler(enabled=(dev == "cuda"))
    mejor, estado, mejor_ep, hist = (-1.0, -float("inf")), None, 0, []
    for ep in range(epocas):
        m.train(); suma, n = 0.0, 0
        for x, y in dl["train"]:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(device_type=dev, enabled=(dev == "cuda")):
                loss = F.cross_entropy(m(x).float(), y, weight=w)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            suma += loss.item() * len(y); n += len(y)
        sch.step()
        if ep == 0 and dev == "cuda":
            torch.cuda.reset_peak_memory_stats()   # la 1.ª época incluye el autotuning de cuDNN
        r = metricas(*predecir(m, dl["val"], dev))
        hist.append({"epoca": ep + 1, "loss_train": suma / n, "val_loss": r["nll"],
                     "val_acc": r["acc"], "val_f1_macro": r["f1_macro"]})
        clave = (r["f1_macro"], -r["nll"])
        if clave > mejor:
            mejor, mejor_ep = clave, ep + 1
            estado = {k: v.detach().clone() for k, v in m.state_dict().items()}
        print(f"seed {seed} ep {ep+1:3d} loss {suma/n:.3f} val loss {r['nll']:.3f} acc {r['acc']:.3f} "
              f"F1 {r['f1_macro']:.3f}", flush=True)
    print(f"seed {seed}: mejor época {mejor_ep} (val F1 {mejor[0]:.3f}, val loss {-mejor[1]:.3f})", flush=True)
    m.load_state_dict(estado)
    return m, metricas(*predecir(m, dl["test"], dev)), hist, mejor_ep


def reporte_por_clase(cm, archivo):
    import pandas as pd
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    pd.DataFrame({"clase": CLASES, "soporte": cm.sum(1), "precision": prec, "recall": rec, "f1": f1}).round(3).to_csv(archivo, index=False)


# ------------------------------------------------------------------ figuras
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def graficas_datos(datos):
    plt = _plt()
    S = tamano(datos); c0 = S // 2
    cnt = {s: np.bincount(datos[s][1], minlength=N_CLASES) for s in datos}
    x, w = np.arange(N_CLASES), 0.28
    plt.figure(figsize=(9, 3.2))
    for i, s in enumerate(("train", "val", "test")):
        plt.bar(x + (i - 1) * w, cnt[s], w, label=s)
    plt.xticks(x, CLASES, rotation=45, ha="right"); plt.ylabel("volúmenes"); plt.legend(); plt.tight_layout()
    plt.savefig("fig_distribucion_clases.png", dpi=200); plt.close()

    X, y = datos["train"]
    fig, ax = plt.subplots(3, N_CLASES, figsize=(16, 4.8))
    for c in range(N_CLASES):
        idx = np.where(y == c)[0]
        for r in range(3):
            ax[r, c].axis("off")
        if len(idx) == 0:
            continue
        v = X[idx[0]]
        for r, sl in enumerate((v[c0, :, :], v[:, c0, :], v[:, :, c0])):
            ax[r, c].imshow(sl, cmap="gray")
        ax[0, c].set_title(CLASES[c], fontsize=8)
    plt.tight_layout(); plt.savefig("fig_muestra_cortes.png", dpi=200); plt.close()


def figura_curvas(historiales):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
    for seed, h in historiales.items():
        e = [r["epoca"] for r in h]
        ax[0].plot(e, [r["loss_train"] for r in h], label=f"semilla {seed}")
        ax[1].plot(e, [r["val_loss"] for r in h], label=f"semilla {seed}")
        ax[2].plot(e, [r["val_f1_macro"] for r in h], label=f"semilla {seed}")
    ax[0].set_xlabel("época"); ax[0].set_ylabel("pérdida de entrenamiento")
    ax[1].set_xlabel("época"); ax[1].set_ylabel("entropía cruzada en validación")
    ax[2].set_xlabel("época"); ax[2].set_ylabel("F1 macro en validación")
    for a in ax:
        a.legend()
    plt.tight_layout(); plt.savefig("fig_curvas.png", dpi=200); plt.close()


def figura_confusion(cm):
    plt = _plt()
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    plt.figure(figsize=(7, 6)); plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(N_CLASES):
        for j in range(N_CLASES):
            if cm[i, j]:
                plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=6, color="white" if cmn[i, j] > 0.5 else "black")
    plt.xticks(range(N_CLASES), CLASES, rotation=60, ha="right", fontsize=7); plt.yticks(range(N_CLASES), CLASES, fontsize=7)
    plt.xlabel("predicha"); plt.ylabel("real"); plt.colorbar(); plt.tight_layout()
    plt.savefig("fig_confusion.png", dpi=200); plt.close()


def figura_predicciones(m, datos):
    """Corte axial central de una muestra de prueba por clase, con la clase real y la predicha."""
    plt = _plt()
    dev = next(m.parameters()).device.type
    P, Y = predecir(m, loaders(datos, 32)["test"], dev)
    pred = P.argmax(1)
    rng = np.random.default_rng(0)
    idx = [rng.choice(np.where(Y == c)[0]) for c in range(N_CLASES) if np.any(Y == c)]
    X, c0 = datos["test"][0], tamano(datos) // 2
    fig, ax = plt.subplots(2, 6, figsize=(13, 5))
    for a in ax.ravel():
        a.axis("off")
    for a, i in zip(ax.ravel(), idx):
        a.imshow(X[i][c0], cmap="gray")
        a.set_title(f"real: {CLASES[Y[i]]}\npred: {CLASES[pred[i]]} ({P[i].max():.2f})", fontsize=8,
                    color="green" if pred[i] == Y[i] else "red")
    plt.tight_layout(); plt.savefig("fig_predicciones.png", dpi=200); plt.close()


# ------------------------------------------------------------------ programa principal
def obtener_ruta(npz, size):
    if npz:
        return npz
    import os
    from medmnist import OrganMNIST3D
    OrganMNIST3D(split="train", download=True, size=size)
    return os.path.expanduser(f"~/.medmnist/organmnist3d{'' if size == 28 else f'_{size}'}.npz")


def main():
    import argparse, time
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None, help="ruta al .npz (si no, se descarga con medmnist)")
    ap.add_argument("--tamano", type=int, default=64, choices=(28, 64))
    ap.add_argument("--epocas", type=int, default=100)
    ap.add_argument("--semillas", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    datos = cargar_npz(obtener_ruta(a.npz, a.tamano))
    assert tamano(datos) == a.tamano, f"el archivo es de {tamano(datos)}³, se pidió {a.tamano}³"
    resumen(datos)
    graficas_datos(datos)
    x, _ = next(iter(loaders(datos, a.batch)["train"]))
    print("lote:", tuple(x.shape), "| parámetros:", n_param(ResNet3D()))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    filas, cms, hists, modelos = [], {}, {}, {}
    for seed in a.semillas:
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t = time.time()
        m, r, h, mejor_ep = entrenar(datos, epocas=a.epocas, batch=a.batch, seed=seed, dev=dev)
        fila = dict(tamano=a.tamano, seed=seed, acc=r["acc"], f1_macro=r["f1_macro"], auc_ovr=r["auc_ovr"],
                    mejor_epoca=mejor_ep, s_por_epoca=(time.time() - t) / a.epocas)
        if dev == "cuda":
            fila["mem_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
        print(fila, flush=True)
        filas.append(fila); cms[seed] = r["cm"]; hists[seed] = h; modelos[seed] = m
        # se guarda todo después de cada semilla para no perder nada si la sesión se corta
        pd.DataFrame(filas).to_csv("resultados.csv", index=False)
        pd.DataFrame(h).to_csv(f"historial_seed{seed}.csv", index=False)
        reporte_por_clase(r["cm"], f"metricas_por_clase_seed{seed}.csv")
        torch.save(m.state_dict(), f"modelo_seed{seed}.pt")
        np.savez("matrices_confusion.npz", **{f"seed{k}": v for k, v in cms.items()})

    df = pd.DataFrame(filas)
    print(df.drop(columns=["tamano", "seed"]).agg(["mean", "std"]).round(3).to_string())
    s0 = a.semillas[0]
    figura_curvas(hists)
    figura_confusion(cms[s0])
    figura_predicciones(modelos[s0], datos)


if __name__ == "__main__":
    main()
