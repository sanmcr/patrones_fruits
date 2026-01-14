#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import random
from collections import defaultdict

# ----------------------------
# Configuración
# ----------------------------
DATASET_DIR = Path("fruits")      # carpeta raíz del dataset
SEED = 42                         # para que el split sea reproducible
TRAIN_RATIO = 0.80                # 80% train
# El 20% restante se parte a la mitad: 10% val, 10% test
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

OUT_TRAIN = Path("train.txt")
OUT_VAL = Path("validation.txt")
OUT_TEST = Path("test.txt")


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def list_class_folders(dataset_dir: Path):
    """Devuelve carpetas hijas (clases) dentro de dataset_dir."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de dataset: {dataset_dir.resolve()}")

    folders = [p for p in dataset_dir.iterdir() if p.is_dir()]
    folders.sort()
    if not folders:
        raise RuntimeError(f"No se han encontrado subcarpetas (clases) dentro de: {dataset_dir.resolve()}")
    return folders


def stratified_split(paths, rng: random.Random, train_ratio: float):
    """
    Dado una lista de rutas de UNA clase:
    - 80% -> train
    - el resto -> val y test, mitad y mitad
    """
    paths = list(paths)
    rng.shuffle(paths)

    n_total = len(paths)
    if n_total < 3:
        # Con poquísimas imágenes no tiene sentido partir en 3 splits bien.
        # Aun así, hacemos algo razonable: 1 train y el resto se reparte.
        n_train = max(1, int(round(n_total * train_ratio)))
    else:
        n_train = int(round(n_total * train_ratio))

    # Asegurar límites
    n_train = min(max(n_train, 1), n_total - 2) if n_total >= 3 else min(n_train, n_total)

    train = paths[:n_train]
    rest = paths[n_train:]

    # dividir resto en dos mitades lo más igualadas posible
    mid = len(rest) // 2
    val = rest[:mid]
    test = rest[mid:]

    return train, val, test


def write_txt(out_path: Path, samples):
    """
    samples: lista de tuplas (ruta_relativa, etiqueta)
    """
    with out_path.open("w", encoding="utf-8") as f:
        for rel_path, label in samples:
            f.write(f"{rel_path.as_posix()} {label}\n")


def main():
    rng = random.Random(SEED)

    class_folders = list_class_folders(DATASET_DIR)

    train_samples = []
    val_samples = []
    test_samples = []

    counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "total": 0})

    # 1) Split estratificado por carpeta/clase
    for class_dir in class_folders:
        label = class_dir.name  # etiqueta = nombre de carpeta
        imgs = [p for p in class_dir.rglob("*") if is_image_file(p)]
        imgs.sort()

        if not imgs:
            print(f"[AVISO] Clase vacía: {label}")
            continue

        tr, va, te = stratified_split(imgs, rng, TRAIN_RATIO)

        counts[label]["train"] = len(tr)
        counts[label]["val"] = len(va)
        counts[label]["test"] = len(te)
        counts[label]["total"] = len(imgs)

        # Guardamos rutas RELATIVAS para que no dependan de tu PC
        for p in tr:
            train_samples.append((p.relative_to(Path(".")), label))
        for p in va:
            val_samples.append((p.relative_to(Path(".")), label))
        for p in te:
            test_samples.append((p.relative_to(Path(".")), label))

    # 2) Barajar el global (para mezclar clases en los .txt)
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    # 3) Escribir ficheros
    write_txt(OUT_TRAIN, train_samples)
    write_txt(OUT_VAL, val_samples)
    write_txt(OUT_TEST, test_samples)

    # 4) Resumen
    print("\n=== SPLIT COMPLETADO ===")
    total_train = len(train_samples)
    total_val = len(val_samples)
    total_test = len(test_samples)
    total_all = total_train + total_val + total_test

    print(f"Train: {total_train} imágenes -> {OUT_TRAIN}")
    print(f"Val:   {total_val} imágenes -> {OUT_VAL}")
    print(f"Test:  {total_test} imágenes -> {OUT_TEST}")
    print(f"Total: {total_all} imágenes\n")

    print("Por clase:")
    for label in sorted(counts.keys()):
        c = counts[label]
        print(f"- {label}: total={c['total']}, train={c['train']}, val={c['val']}, test={c['test']}")


if __name__ == "__main__":
    main()

