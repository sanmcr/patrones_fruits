#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


# ----------------------------
# Configuración
# ----------------------------
TRAIN_TXT = Path("train.txt")
VAL_TXT = Path("validation.txt")

MODEL_OUT = Path("modelo_frutas.keras")     # formato recomendado Keras
LABEL_MAP_OUT = Path("label_map.json")

IMG_SIZE = (160, 160)      # tamaño moderado para ir rápido y no inflar el modelo
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def read_txt_list(txt_path: Path):
    """
    Lee un fichero tipo:
        ruta etiqueta
    Devuelve:
        paths: lista[str]
        labels: lista[str]
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"No existe: {txt_path.resolve()}")

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


def build_label_map(all_label_strs):
    """
    Crea un mapeo estable: label -> index, index -> label
    """
    unique = sorted(set(all_label_strs))
    label_to_idx = {lab: i for i, lab in enumerate(unique)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}
    return label_to_idx, idx_to_label


def decode_and_resize(path, label_idx):
    """
    Carga imagen, decodifica, normaliza y redimensiona.
    """
    img_bytes = tf.io.read_file(path)

    # decode_image soporta jpg/png/etc. y devuelve float/uint8 según convert.
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")

    return img, label_idx


def make_dataset(paths, label_idxs, training: bool):
    """
    Crea tf.data.Dataset eficiente.
    """
    ds = tf.data.Dataset.from_tensor_slices((paths, label_idxs))

    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE)

    if training:
        # Augmentación básica y barata (sin flip vertical por frutas, queda raro)
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(0.06, seed=SEED),
            tf.keras.layers.RandomZoom(0.10, seed=SEED),
            tf.keras.layers.RandomContrast(0.10, seed=SEED),
        ])

        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def build_model(num_classes: int):
    """
    CNN pequeñita y efectiva para 6 clases.
    """
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    # Reproducibilidad decente
    tf.keras.utils.set_random_seed(SEED)

    # 1) Leer listas
    train_paths, train_labels = read_txt_list(TRAIN_TXT)
    val_paths, val_labels = read_txt_list(VAL_TXT)

    # 2) Construir label map usando train+val (mismo conjunto de clases)
    label_to_idx, idx_to_label = build_label_map(train_labels + val_labels)
    num_classes = len(label_to_idx)

    if num_classes < 2:
        raise RuntimeError("Solo se ha detectado 1 clase. Revisa los .txt y las carpetas.")

    # 3) Convertir rutas a tensores (string) y labels a índices
    train_label_idxs = np.array([label_to_idx[l] for l in train_labels], dtype=np.int32)
    val_label_idxs = np.array([label_to_idx[l] for l in val_labels], dtype=np.int32)

    train_ds = make_dataset(train_paths, train_label_idxs, training=True)
    val_ds = make_dataset(val_paths, val_label_idxs, training=False)

    # 4) Modelo
    model = build_model(num_classes)
    model.summary()

    # 5) Callbacks: guardar mejor modelo y parar si no mejora
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_OUT.as_posix(),
            monitor="val_accuracy",
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]

    # 6) Entrenar
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # 7) Guardar label map (muy útil para finalproduct.py)
    with LABEL_MAP_OUT.open("w", encoding="utf-8") as f:
        json.dump(
            {"label_to_idx": label_to_idx, "idx_to_label": {str(k): v for k, v in idx_to_label.items()}},
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print(f"Modelo guardado en: {MODEL_OUT.resolve()}")
    print(f"Mapa de etiquetas guardado en: {LABEL_MAP_OUT.resolve()}")


if __name__ == "__main__":
    main()
