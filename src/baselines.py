"""Экстрактивные baseline-ы для суммаризации."""


def first_sentence(text):
    """Первое предложение длиннее 10 символов."""
    for sent in text.split("."):
        s = sent.strip()
        if len(s) > 10:
            return s + "."
    return text[:200]


def last_sentence(text):
    """Последнее предложение длиннее 10 символов."""
    parts = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
    return parts[-1] + "." if parts else text[-200:]


def first_last(text):
    """Первое + последнее предложение."""
    parts = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
    if len(parts) >= 2:
        return parts[0] + ". " + parts[-1] + "."
    return parts[0] + "." if parts else text[:300]
