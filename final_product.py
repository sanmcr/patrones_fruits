import json
from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf


# ============================
# Configuración
# ============================
TEST_TXT = Path("test.txt")
MODEL_PATH = Path("modelo_frutas.keras")
LABEL_MAP_PATH = Path("label_map.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

OUT_PRED = Path("etiquetas_estimadas.txt")
OUT_CM = Path("matriz_confusion.txt")


def read_list(txt_path: Path):
    paths, labels = [], []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            paths.append(parts[0])
            labels.append(parts[1])
    return paths, labels


def decode_img(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def load_example(path, label_idx):
    img_bytes = tf.io.read_file(path)
    img = decode_img(img_bytes)
    return img, label_idx


def make_dataset(paths, label_idxs):
    ds = tf.data.Dataset.from_tensor_slices((paths, label_idxs))
    ds = ds.map(load_example, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def main():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"No existe el modelo: {MODEL_PATH}")
    if not LABEL_MAP_PATH.exists():
        raise RuntimeError(f"No existe el label map: {LABEL_MAP_PATH}")
    if not TEST_TXT.exists():
        raise RuntimeError(f"No existe el fichero de test: {TEST_TXT}")

    # 1) Cargar label_map
    with LABEL_MAP_PATH.open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    label_to_idx = label_map["label_to_idx"]
    idx_to_label = {int(k): v for k, v in label_map["idx_to_label"].items()}

    # 2) Leer test
    test_paths, test_labels = read_list(TEST_TXT)
    y_true = np.array([label_to_idx[l] for l in test_labels], dtype=np.int32)

    # 3) Dataset
    test_ds = make_dataset(test_paths, y_true)

    # 4) Cargar modelo
    model = tf.keras.models.load_model(MODEL_PATH)

    # 5) Predicciones
    probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probs, axis=1).astype(np.int32)

    # 6) Accuracy
    acc = float(np.mean(y_pred == y_true))
    print(f"\n Accuracy test: {acc:.4f}")

    # 7) Matriz de confusión (a mano, sin sklearn)
    num_classes = len(idx_to_label)
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # 8) Guardar predicciones (ruta, label_real, label_pred)
    with OUT_PRED.open("w", encoding="utf-8") as f:
        for path, t, p in zip(test_paths, y_true, y_pred):
            f.write(f"{path} {idx_to_label[int(t)]} {idx_to_label[int(p)]}\n")
    print(f" Etiquetas estimadas guardadas en: {OUT_PRED}")

    # 9) Guardar matriz de confusión
    with OUT_CM.open("w", encoding="utf-8") as f:
        f.write("Clases (idx -> label):\n")
        for i in range(num_classes):
            f.write(f"  {i}: {idx_to_label[i]}\n")
        f.write("\nMatriz de confusión (filas=real, cols=pred):\n")
        f.write(np.array2string(cm))
        f.write(f"\n\nAccuracy: {acc:.4f}\n")
    print(f" Matriz de confusión guardada en: {OUT_CM}")

    # (Opcional) distribución de predicciones por curiosidad
    pred_counts = Counter(y_pred.tolist())
    print("\nDistribución de predicciones:")
    for k in sorted(pred_counts.keys()):
        print(f"  {k} ({idx_to_label[k]}): {pred_counts[k]}")


if __name__ == "__main__":
    main()
