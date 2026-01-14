import json
from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf
import random

# ----------------------------
# Configuración
# ----------------------------
TRAIN_TXT = Path("train.txt")
VAL_TXT = Path("validation.txt")

MODEL_OUT = Path("modelo_frutas.keras")
LABEL_MAP_OUT = Path("label_map.json")

IMG_SIZE = (224, 224)      # subimos resolución para captar mejor detalles (fresh vs rotten)
BATCH_SIZE = 32
EPOCHS = 25
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def read_list(txt_path: Path):
    """
    Lee un fichero .txt donde cada línea es:
        ruta/a/imagen.jpg etiqueta
    Devuelve: (paths, labels) como listas de strings.
    """
    paths, labels = [], []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            img_path = parts[0]
            label = parts[1]
            paths.append(img_path)
            labels.append(label)
    return paths, labels


def decode_img(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def load_example(path, label):
    img_bytes = tf.io.read_file(path)
    img = decode_img(img_bytes)
    return img, label


def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    return img, label


def make_dataset(paths, labels, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_example, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def build_model(num_classes: int):
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    # 1) Leer listas
    train_paths, train_labels = read_list(TRAIN_TXT)
    val_paths, val_labels = read_list(VAL_TXT)

    if len(train_paths) == 0:
        raise RuntimeError("train.txt está vacío o no se ha podido leer correctamente.")
    if len(val_paths) == 0:
        raise RuntimeError("validation.txt está vacío o no se ha podido leer correctamente.")

    # 2) Crear label_map desde lo visto en train (y val por seguridad)
    unique_labels = sorted(set(train_labels) | set(val_labels))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx_to_label = {str(i): lab for lab, i in label_to_idx.items()}
    num_classes = len(unique_labels)

    if num_classes < 2:
        raise RuntimeError("Solo se ha detectado 1 clase. Revisa los .txt y las carpetas.")

    # 3) Convertir rutas a tensores (string) y labels a índices
    train_label_idxs = np.array([label_to_idx[l] for l in train_labels], dtype=np.int32)
    val_label_idxs = np.array([label_to_idx[l] for l in val_labels], dtype=np.int32)

    train_ds = make_dataset(train_paths, train_label_idxs, training=True)
    val_ds = make_dataset(val_paths, val_label_idxs, training=False)

    # 4) Modelo
    model = build_model(num_classes=num_classes)
    model.summary()

    # 5) Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_OUT),
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
    # 6) Balanceo por pesos de clase (ayuda si hay clases con menos ejemplos)
    counts = Counter(train_label_idxs.tolist())
    n_total = len(train_label_idxs)
    class_weight = {cls: n_total / (num_classes * cnt) for cls, cnt in counts.items()}
    print("Class weights:", class_weight)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight
    )

    # 7) Guardar label map (muy útil para finalproduct.py)
    with LABEL_MAP_OUT.open("w", encoding="utf-8") as f:
        json.dump({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, f, indent=2, ensure_ascii=False)

    print(f"\n Modelo guardado en: {MODEL_OUT}")
    print(f"Label map guardado en: {LABEL_MAP_OUT}")


if __name__ == "__main__":
    main()
