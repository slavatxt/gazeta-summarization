"""Загрузка и подготовка данных Gazeta."""
import os
import pandas as pd


def find_data_path(base="/kaggle/input"):
    """Рекурсивный поиск gazeta_train.jsonl."""
    for root, dirs, files in os.walk(base):
        if "gazeta_train.jsonl" in files:
            return root
    raise FileNotFoundError(
        f"gazeta_train.jsonl не найден в {base}. "
        "Добавьте датасет gazeta-summaries через Add Input."
    )


def load_gazeta(data_path=None):
    """Загрузить train/val/test датафреймы."""
    if data_path is None:
        data_path = find_data_path()
    train = pd.read_json(os.path.join(data_path, "gazeta_train.jsonl"), lines=True)
    val = pd.read_json(os.path.join(data_path, "gazeta_val.jsonl"), lines=True)
    test = pd.read_json(os.path.join(data_path, "gazeta_test.jsonl"), lines=True)
    return train, val, test
