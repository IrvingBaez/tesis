# Tesis: Audio-Visual Diarization (AVD)

Este proyecto implementa un sistema de **diarización audiovisual**: dado un video, determina quién habló y cuándo. El modelo central es una versión extendida de AVR-Net que incorpora mecanismos de atención propia (self-attention) y atención cruzada (cross-attention) entre modalidades de video y audio.

El dataset utilizado es **AVA-AVD** (Active Speaker Detection / Audio Visual Diarization), particionado en train/val/test según los archivos en `dataset/split/`.

---

## Punto de entrada y flujo general

```
run.sh
  └─ python3 model/experiments.py
        └─ train_lightning_avd(**train_params)
              └─ model/third_party/avr_net/train_lightning.py::main()
                    ├─ initialize_arguments()   # convierte train_params → args namespace
                    └─ train(args)
                          ├─ create_dataset(args)
                          │     └─ feature_extraction.py::main()  # extrae y cachea features
                          └─ pl.Trainer.fit() / .validate() / .test()
```

### `run.sh`
Activa el entorno conda `tesis`, configura variables de entorno (PYTHONPATH, CUDA, NCCL) y ejecuta `model/experiments.py`. La variable `CUDA_VISIBLE_DEVICES=0` fuerza el uso de la GPU 0.

### `model/experiments.py` y `model/runs.py`
`model/runs.py` registra las corridas del plan de experimentos: `BASE` tiene la configuración común y `RUNS` solo las diferencias de cada ID (E1, E2, …). `experiments.py` recibe los IDs y los ejecuta uno tras otro:

```bash
./run.sh E4 E1 E2        # corre en ese orden
./run.sh                 # sin IDs: solo imprime la tabla resumen
./run.sh P1 P2 P3 P4 P5  # pilotos: 10 % de train, 25 % de val, 3 épocas
```

Los pilotos (`P*` en `runs.py`) comprueban que una configuración aprende desde cero antes de lanzar las corridas completas. El primer E4 completo (SGD, lr 5e-4, sin momentum, sin barajar) se quedó en nivel de azar; está archivado en `lightning_logs/archive/`.

- **Recuperación** (`model/run_summary.py::run_state`): al relanzar el mismo comando, cada corrida terminada (`done.json`) se salta; una detenida con `last.ckpt` continúa desde la última época completa (config original, mismo folder); una que murió antes de cerrar su primera época se borra y empieza de nuevo.
- **Ctrl+C**: Lightning lo captura y regresa sin excepción; `experiments.py` revisa `trainer.interrupted` y detiene la cola.
- Si una corrida falla con excepción, se sigue con la siguiente y al final se reportan las fallidas.
- **Tabla resumen**: siempre se imprime al final (también tras Ctrl+C o errores) y se guarda en `lightning_logs/summary.txt`. Se arma leyendo los archivos de cada corrida, así que es completa aunque las corridas se hayan hecho en varias ejecuciones.
- **Figuras** (`model/plots.py`): histogramas de scores, DER contra umbral del AHC y curvas por época, en PDF para la tesis (`../ThesisDoc/images/results` por defecto). Acepta IDs o carpetas de corridas: `python model/plots.py histograms E2 E3 E5`.

Cada corrida escribe en `model/third_party/avr_net/checkpoints/lightning_logs/<ID>/` (TensorBoard, `config.json` con hiperparámetros y entorno, checkpoints y los resultados descritos abajo).

---

## Parámetros de `train_params` (experiments.py)

### Datos
| Parámetro | Tipo | Descripción |
|---|---|---|
| `video_proportion` | float (0,1] | Fracción de videos de train a usar |
| `val_video_proportion` | float (0,1] | Fracción de videos de val a usar |
| `aligned` | bool | Usar faces alineadas (`aligned_tracklets/`) en lugar de `tracklets/` |
| `balanced` | bool | Balancear pares positivos/negativos en train |
| `max_frames` | int | Número de frames por utterance guardados en los `.pckl` de features |
| `db_video_mode` | str | Modo de reducción de frames en extracción: `pick_first`, `pick_random`, `keep_all`, `average` |
| `checkpoint` | str o None | Ruta a un `.ckpt` del que se cargan los pesos (o que se evalúa) |
| `relation_init` | str o None | Pesos preentrenados de la relation layer. `None` = entrenar desde cero |
| `run_name` | str o None | ID de la corrida; nombre de la carpeta de logs |
| `resume` | bool | Continuar `run_name` desde su `last.ckpt` (lo decide `experiments.py`) |
| `seed` | int | Semilla para `pl.seed_everything` |
| `task` | str | `'train'`, `'val'`, o `'test'` |

### Arquitectura
| Parámetro | Tipo | Opciones | Descripción |
|---|---|---|---|
| `self_attention` | str | `'class_token'`, `'temporal'`, `'pick_first'` | Cómo agregar N frames de una utterance en 1 |
| `self_attention_dropout` | float | — | Dropout del Transformer (solo aplica a `class_token`) |
| `cross_attention` | str | `'concat'`, `'fusion'` | Cómo fusionar features de video y audio |
| `fine_tunning` | bool | — | Si True, congela todo excepto `relation_layer.fc` (última capa) |

### Hiperparámetros
| Parámetro | Tipo | Descripción |
|---|---|---|
| `loss_fn` | str | `'bce'`, `'mse'`, `'contrastive'` |
| `pos_margin` | float [0, 0.5] | Margen positivo en contrastive loss: distancias menores no generan pérdida |
| `neg_margin` | float [0.5, 1.0] | Margen negativo: distancias mayores no generan pérdida |
| `ahc_threshold` | float | Umbral de similitud para el clustering AHC en validación (defecto original: 0.14) |
| `optimizer` | str | `'sgd'` o `'adam'` |
| `learning_rate` | float | LR base |
| `momentum` | float | Momentum de SGD (ignorado con Adam) |
| `weight_decay` | float | Regularización L2 |
| `step_size` | int | Pasos de época del scheduler `StepLR` |
| `gamma` | float | Factor de decaimiento del scheduler |
| `epochs` | int | Número fijo de épocas (se pasa como `min_epochs` y `max_epochs` al Trainer) |
| `frozen_epochs` | int | Épocas con la relation layer congelada al inicio |
| `disable_pb` | bool | Ocultar barras de progreso tqdm |

---

## Flujo detallado de `train_lightning.py`

### `initialize_arguments()`
- Convierte el dict de kwargs a un `argparse.Namespace` mediante `argparse_helper()` (en `model/util.py`).
- Construye `train_dataset_config` y `val_dataset_config` usando `get_path()` para resolver rutas del dataset.
- Hace `assert` de que todos los paths existen — **primer punto de falla** si el dataset no está montado correctamente.
- Configura `args.sys_path = 'model/third_party/avr_net/features'` y `args.checkpoint_dir`.

### `train()`
1. Llama a `create_dataset(args)` que internamente:
   - Ejecuta `feature_extraction.py::main()` (extrae features y las guarda en `.pckl` por video)
   - Crea `train_loader` (batch=64, workers=11) y `eval_loader` (batch=512, workers=2) con `ClusteringDataset`
   - Train se baraja cada época (orden fijado por la semilla) y precarga en RAM las features de todos sus videos (`preload=True`, ~12 GB en modo `average`, ~17 GB con los workers). Sin precarga, el acceso aleatorio cargaría un `.pckl` distinto casi en cada par. Validación y test van en orden, con caché de un video
   - `training='episodic'` (E0): `tools/episodic_dataset.py` genera de antemano, con la semilla, los episodios del AVR-Net original para todas las épocas (20 hablantes × 2 *utterances*, 400 pares *query* × *support*). Cada época tiene tantos pares como el entrenamiento balanceado, y se barajan y agrupan en *batches* de 64 como cualquier otro par; `EpochSampler` toma los de la época actual. Solo cambian los pares, no el proceso de entrenamiento
   - `shuffle_block_videos=N` (E8, E9): para *features* que no caben en RAM, no precarga; `BlockShuffleSampler` baraja dentro de bloques de N videos y cada *worker* guarda un bloque
   - `frame_selection='center'` (E8, E9): usa los `max_frames` frames centrales de cada *utterance* en lugar de los primeros; sus *features* van a `<max_frames>_frames_center_<modo>`
2. Extrae `utterance_counts`, `starts`, `ends` del dataset para usar en métricas de validación.
3. Crea `Lightning_Attention_AVRNet` con `clean_up_args(args)` (filtra args no serializables).
4. Si hay `checkpoint`, carga `state_dict` directamente.
5. Configura `pl.Trainer` con:
   - `ModelCheckpoint(monitor="der/val", mode="min", save_last=True)` (no hay EarlyStopping)
6. Ejecuta `trainer.fit()`, `trainer.validate()` o `trainer.test()` según `args.task`.
7. Al terminar `fit`, `score_best_epoch_on_train()` calcula una sola vez el DER de train sobre todos los pares, con el checkpoint y el umbral del AHC de la mejor época de validación. Deja `train/summary.json` y `der/train`, y después escribe `done.json`. No se calcula por época porque tarda ~1 h 40 min sobre los 12.6 M pares.

---

## Arquitectura del modelo (`attention_avr_net.py`)

```
Entrada: video [B, 2*N, 512, 7, 7], audio [B, 2, 256, 7, 7], task [2, B]
│
├─ BatchNorm2d(256) sobre el audio
│
├─ self_attention(video_a)  →  [B, 1, 512, 7, 7]   # agrupa N frames → 1
├─ self_attention(video_b)  →  [B, 1, 512, 7, 7]
│
├─ cross_attention(video_a, audio_a)  →  [B, 1, 768, 7, 7]  # funde V+A
├─ cross_attention(video_b, audio_b)  →  [B, 1, 768, 7, 7]
│
└─ relation_layer(cat(feats_a, feats_b), task)  →  score [B, 1]  (sigmoid)
```

### Self-attention (`tools/attention.py`)
- **`pick_first`** (o cualquier valor no reconocido): toma el frame 0. Sin parámetros entrenables.
- **`class_token`**: Transformer 1-capa con class token. El class token agrega la secuencia.
- **`temporal`**: TemporalAttentionPooling — pool espacial → linear → softmax → suma ponderada de frames.

### Cross-attention (`tools/attention.py`)
- **`concat`** (o cualquier valor no reconocido): concatena canales de video y audio. Sin parámetros entrenables.
- **`fusion`**: FusionCrossAttention — cross-attention bidireccional entre las dos modalidades.

### RelationLayer (`models/relation_layer.py`)
- Se entrena desde cero salvo que se dé `relation_init` (antes cargaba siempre `weights/best_relation.ckpt`).
- Arquitectura ResNet-like (Bottleneck ×2) + FC con sigmoid.
- `task_token`: embedding de tamaño (4, 1536) — codifica la visibilidad del par de utterances (visible_a, visible_b → 4 combinaciones posibles).
- Siempre se inicializa congelado; se descongela tras `frozen_epochs` épocas (a menos que `fine_tunning=True`).

---

## Pipeline de métricas de validación (`on_validation_epoch_end`)

**El DER (Diarization Error Rate) es la métrica principal. Valores más bajos son mejores. Un DER de 0 significa diarización perfecta.**

1. Agrupa todas las predicciones (scores por par de utterances) en matrices de similitud por video.
2. Guarda la matriz en `{log_dir}/similarities.pth`.
3. **`write_rttms()`** (`tools/write_rttms.py`):
   - Aplica **AHC clustering** (`tools/ahc_cluster.py`) barriendo todos los valores de `AHC_THRESHOLDS` sobre la matriz de similitud.
   - Fusiona segmentos adyacentes del mismo hablante (`merge_frames`).
   - Escribe archivos `.rttm` por video, más un listado `.out`.
4. **`score_avd()`** (`avd/score_avd.py`):
   - Compara las RTTMs generadas contra las de ground truth usando `dscore` (DER con collar=0.25s).
5. Loguea `der/val_<N>` para cada threshold y `der/val` = mínimo del sweep → usado por `ModelCheckpoint`. Del mejor umbral también loguea el desglose (`der_miss/val`, `der_fa/val`, `der_confusion/val`, calculado con el mismo `md-eval-22.pl` que el DER) y el umbral (`ahc_threshold/val`).
6. Artefactos que se conservan en `lightning_logs/<ID>/`:
   - `val_<época>/`: `similarities.pth`, las RTTMs de todos los umbrales y `summary.json` (umbral elegido, DER, desglose, F1/precision/recall de los pares a 0.5, resultados por umbral y `epoch_seconds`). Si una época se repite al reanudar, su carpeta se sobrescribe. El *sanity check* de Lightning no deja nada.
   - `test/` (con `task='test'`): lo mismo para test. El umbral nunca se elige en test: `der/test` y el `summary.json` usan el de la mejor época de validación de la corrida (`best_val_summary`, mismo criterio que `ModelCheckpoint`), o `ahc_threshold` si la corrida no tiene validación. Evaluar val o test no reescribe el `config.json` de entrenamiento.
   - `done.json`: se escribe solo si `fit` terminó todas las épocas.

---

## Extracción de features (`feature_extraction.py`)

- Genera archivos `.pckl` por video en `model/third_party/avr_net/features/{max_frames}_frames[_aligned]_{db_video_mode}/{split}/{video_id}.pckl`.
- **Si el `.pckl` ya existe, se salta** — para re-extraer hay que borrar los archivos.
- Backbones congelados: `AudioEncoder` (desde `weights/backbone_audio.pth`) y `VideoEncoder` (desde `weights/backbone_faces.pth`).
- `db_video_mode` controla cómo se reducen N frames a 1 en esta etapa (excepto `keep_all`, que guarda todos).

### Estructura de un `.pckl`
```python
{
    'feat_video': Tensor [U, N, 512, 7, 7],  # U = utterances, N = frames
    'feat_audio': Tensor [U, 1, 256, 7, 7],
    'targets':    Tensor [U],                 # speaker ID numérico
    'visible':    Tensor [U],                 # 1 si utterance tiene cara visible
    'start':      Tensor [U],                 # tiempo de inicio en segundos
    'end':        Tensor [U],
}
```

---

## Paths del dataset

Todos los paths se construyen mediante `model/util.py::get_path()`:

| Variable | Path |
|---|---|
| Waves (audio) | `dataset/waves/dihard18/` |
| RTTMs (GT diarización) | `dataset/avd/ground_truth/predictions/` |
| Labs (GT VAD) | `dataset/vad/ground_truth/predictions/` |
| Frames (caras) | `dataset/asd/ground_truth/tracklets/` o `aligned_tracklets/` |
| Features extraídas | `model/third_party/avr_net/features/` |
| Checkpoints | `model/third_party/avr_net/checkpoints/` |
| Pesos preentrenados | `model/third_party/avr_net/weights/` |

Splits de video IDs: `dataset/split/train.list`, `val.list`, `test.list`.

---

## Guía de troubleshooting por síntoma

### "Path does not exist" / AssertionError en paths
→ `train_lightning.py::initialize_arguments()` líneas 390-397. Verificar que el dataset esté montado y que los paths en `get_path()` existan.

### Error durante la extracción de features
→ `feature_extraction.py`. Revisar que los backbones en `weights/` existan y que el dataset de frames/audio esté completo. Los `.pckl` corruptos se saltan silenciosamente.

### Error en el DataLoader / collator
→ `tools/clustering_dataset.py` y `tools/custom_collator.py`. El dataset indexa pares de utterances de la diagonal superior de la matriz de similitud — cualquier inconsistencia en `utterance_counts` puede causar índices fuera de rango.

### Error de shape en el modelo
→ `attention_avr_net.py::forward()`. Verificar que `max_frames` sea consistente con el `db_video_mode` usado en extracción. Si `db_video_mode='keep_all'`, el tensor de video tiene shape `[B, 2*max_frames, 512, 7, 7]`; si es otro modo, `N=1`.

### Loss NaN o entrenamiento inestable
→ `tools/contrastive_loss.py`. Revisar `pos_margin` y `neg_margin` (deben cumplir `pos_margin < neg_margin`). También revisar `learning_rate` y `weight_decay`.

### DER no mejora
→ `ahc_threshold` en `on_validation_epoch_end()`. El threshold controla la granularidad del clustering: el AHC une clusters mientras su similitud supere el umbral, así que valores bajos → menos clusters (sub-segmentación) y valores altos → más clusters (sobre-segmentación). El valor original del paper era 0.14.

### `frozen_epochs` no tiene efecto
→ `Lightning_Attention_AVRNet::on_train_epoch_start()`. Si `fine_tunning=True`, la relation layer nunca se descongela (solo el FC final queda activo). Ambos flags son mutuamente excluyentes en la práctica.

### Checkpoint no carga correctamente
→ `train()` líneas 422-429. Se carga con `load_state_dict` directo sobre el `state_dict` del `.ckpt`. Si hay desajuste de claves (e.g., cambió la arquitectura de atención), fallará con error de mismatched keys.

### Error de NCCL / GPU
→ Variables de entorno en `run.sh`. `CUDA_VISIBLE_DEVICES=0` fuerza GPU 0. El trainer usa `strategy="auto"` con 1 GPU — si se quiere multi-GPU hay que cambiar `devices` y `strategy`.
