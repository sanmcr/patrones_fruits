#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

OUT_LABELS = Path("etiquetas_estimadas.txt")
OUT_CM = Path("matriz_confusion.txt")

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


# ============================
# Utilidades
# ============================
def read_txt_list(txt_path: Path):
    """
    Lee un fichero tipo:
        ruta etiqueta
    """
    paths, labels = [], []

    with txt_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Línea {i} en {txt_path} no tiene formato 'ruta etiqueta':\n{line}"
                )
            p, lab = parts
            paths.append(p)
            labels.append(lab)

    return paths, labels


def load_label_map(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))

    label_to_idx = data["label_to_idx"]
    idx_to_label = {int(k): v for k, v in data["idx_to_label"].items()}

    class_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    return label_to_idx, idx_to_label, class_names


def decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, IMG_SIZE)
    return img


def make_dataset(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def format_confusion_matrix(cm, class_names):
    name_w = max(len(n) for n in class_names + ["real\\pred"])
    cell_w = max(5, max(len(str(int(x))) for x in cm.flatten()))

    header = " " * (name_w + 1) + " ".join(n.rjust(cell_w) for n in class_names)
    lines = [header]

    for i, name in enumerate(class_names):
        row = " ".join(str(int(cm[i, j])).rjust(cell_w) for j in range(len(class_names)))
        lines.append(name.ljust(name_w) + " " + row)

    return "\n".join(lines)


# ============================
# Main
# ============================
def main():
    # 1) Cargar modelo
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2) Cargar mapa de etiquetas
    label_to_idx, idx_to_label, class_names = load_label_map(LABEL_MAP_PATH)

    # 3) Leer test.txt
    test_paths, test_labels_str = read_txt_list(TEST_TXT)

    y_true = np.array([label_to_idx[l] for l in test_labels_str], dtype=np.int32)

    # 4) Dataset y predicción
    ds = make_dataset(test_paths)
    probs = model.predict(ds, verbose=1)
    y_pred = np.argmax(probs, axis=1).astype(np.int32)

    # 5) Guardar etiquetas_estimadas.txt
    with OUT_LABELS.open("w", encoding="utf-8") as f:
        f.write("ruta etiqueta_real etiqueta_predicha\n")
        for p, yt, yp in zip(test_paths, y_true, y_pred):
            f.write(f"{p} {idx_to_label[int(yt)]} {idx_to_label[int(yp)]}\n")

    # 6) Matriz de confusión (FORMA COMPATIBLE)
    cm = tf.math.confusion_matrix(
        labels=y_true,
        predictions=y_pred,
        num_classes=len(class_names)
    ).numpy()

    accuracy = float(np.mean(y_true == y_pred))

    # 7) Guardar matriz_confusion.txt
    with OUT_CM.open("w", encoding="utf-8") as f:
        f.write("=== MATRIZ DE CONFUSION ===\n")
        f.write("(filas = real, columnas = predicha)\n\n")
        f.write(format_confusion_matrix(cm, class_names))
        f.write("\n\nAccuracy global: {:.4f}\n\n".format(accuracy))

        support = Counter(y_true.tolist())
        f.write("Soporte por clase:\n")
        for i, name in enumerate(class_names):
            f.write(f"- {name}: {support.get(i, 0)}\n")

    print("\n=== FINALPRODUCT COMPLETADO ===")
    print(f"Etiquetas: {OUT_LABELS.resolve()}")
    print(f"Matriz de confusión: {OUT_CM.resolve()}")
    print(f"Accuracy global: {accuracy:.4f}")


if __name__ == "__main__":
    main()
