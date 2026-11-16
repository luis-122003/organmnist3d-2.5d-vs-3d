# Clasificación multiclase de órganos en tomografía 3D: 2.5D frente a 3D sobre OrganMNIST3D

Proyecto final de la Electiva de Ingeniería de Software (Universidad de San Buenaventura Cali).
Autor: Luis Edwar Mosquera Congo.

Clasificación de volúmenes de tomografía computarizada en 11 órganos con el dataset
[OrganMNIST3D](https://medmnist.com/) (MedMNIST v2 / MedMNIST+, CC BY 4.0; 971 / 161 / 610 volúmenes).

- **Modelo principal:** ResNet 3D (2,07 M de parámetros) entrenada a 64×64×64.
- **Comparación:** la misma ResNet 3D y una ResNet 2.5D de parámetros casi iguales (2,15 M) a 28×28×28.
- **Protocolo:** 100 épocas, Adam 1e-3 con decaimiento coseno, lote 32, entropía cruzada ponderada por clase,
  selección del mejor punto por F1 macro en validación (en empate, menor entropía cruzada de validación),
  tres semillas (0, 1 y 2), una sola evaluación en prueba.

## Resultados en prueba (610 volúmenes, media ± desviación de tres semillas, GPU Tesla T4 en Kaggle)

| Modelo | Exactitud | F1 macro | AUC OvR | Memoria GPU (GB) | s/época |
|---|---|---|---|---|---|
| ResNet 3D 64³ | 0,933 ± 0,031 | 0,942 ± 0,031 | 0,997 ± 0,003 | 3,755 ± 0,018 | 27,913 ± 0,084 |
| ResNet 2.5D 28³ | 0,890 ± 0,032 | 0,906 ± 0,028 | 0,994 ± 0,003 | 0,149 ± 0,018 | 0,886 ± 0,025 |
| ResNet 3D 28³ | 0,912 ± 0,023 | 0,922 ± 0,023 | 0,996 ± 0,002 | 0,729 ± 0,017 | 7,222 ± 0,001 |

Los valores salen de `resultados_64/resultados.csv` y `resultados_28/resultados.csv`.

## Contenido

| Carpeta | Qué hay |
|---|---|
| `notebooks/` | `OrganMNIST3D_64_Kaggle.ipynb` (corrida 64³) y `OrganMNIST3D_Kaggle.ipynb` (2.5D vs 3D a 28³), los notebooks ya ejecutados con sus salidas (`*_ejecutado*.ipynb`) y `Clasificar_volumen_OrganMNIST3D.ipynb`, que carga `modelo_seed0.pt` y clasifica cualquier volumen de prueba mostrando las 11 probabilidades (corre en CPU) |
| `codigo/` | `organmnist3d_64.py` (ResNet 3D a 64³), `organmnist3d_completo.py` (2.5D vs 3D a 28³) y `figuras_articulo.py` (Figuras 4 y 5 del informe) |
| `resultados_64/` | Métricas por semilla y por clase, historiales, matrices de confusión, figuras, log, entorno y los pesos `modelo_seed{0,1,2}.pt` |
| `resultados_28/` | Lo mismo para la comparación a 28³ (sin pesos: ese script no los guarda) |
| `figuras_64/` | Figuras del informe generadas con la versión 64×64×64 |

Notebooks ejecutados en Kaggle:
[OrganMNIST3D_64_Kaggle](https://www.kaggle.com/code/luismc15/organmnist3d-64-kaggle) y
[OrganMNIST3D_Kaggle](https://www.kaggle.com/code/luismc15/organmnist3d-kaggle).

## Reproducir

```bash
pip install medmnist torch numpy scikit-learn matplotlib pandas
python codigo/organmnist3d_64.py                 # descarga organmnist3d_64.npz de Zenodo, verifica el MD5 y entrena (3 semillas, ~2,3 h en T4)
python codigo/organmnist3d_completo.py           # comparación 2.5D vs 3D a 28³ (~42 min en T4)
python codigo/organmnist3d_64.py --epocas 1 --semillas 0   # prueba rápida
```

Para cargar un modelo entrenado:

```python
import torch
from codigo.organmnist3d_64 import ResNet3D
m = ResNet3D(); m.load_state_dict(torch.load("resultados_64/modelo_seed0.pt", map_location="cpu")); m.eval()
```

Los datos no se incluyen: los scripts los descargan de Zenodo (registro 10519652) y comprueban el MD5 oficial.
