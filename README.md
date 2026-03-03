# Gazeta Summarization: сравнение методов дообучения LLM

Суммаризация русских новостей из датасета [Gazeta](https://www.kaggle.com/datasets/phoenix120/gazeta-summaries).  
В проекте сравниваются экстрактивные baseline-ы и три подхода к дообучению decoder-only модели **Qwen2.5-0.5B-Instruct** (494M параметров).

## Результаты

| Метод | ROUGE-1 | ROUGE-2 | ROUGE-L | Обучаемые параметры |
|-------|---------|---------|---------|---------------------|
| **Baseline (1-е предложение)** | **0.1833** | **0.0638** | **0.1808** | — |
| Baseline (последнее предл.) | 0.0331 | 0.0123 | 0.0328 | — |
| Baseline (первое + последнее) | 0.1736 | 0.0584 | 0.1705 | — |
| Head-only (lm_head) | 0.1532 | 0.0492 | 0.1486 | 136M (28%) |
| QLoRA + Unsloth (4-bit) | 0.1258 | 0.0376 | 0.1238 | 8.8M (1.7%) |
| Full fine-tuning | 0.0048 | 0.0000 | 0.0048 | 494M (100%) |

## Выводы

1. **Baseline побеждает.** Для новостных текстов первое предложение оказывается сильным baseline-ом (принцип перевернутой пирамиды). Модель 0.5B на 2000 примерах не превосходит этот уровень.
2. **Head-only — лучший из обучаемых подходов.** Обучается только `lm_head` (и tied embeddings, 28% параметров), что снижает риск переобучения.
3. **QLoRA уступает head-only.** При 1.7% обучаемых параметров (LoRA `r=16`) на 2000 примерах емкости недостаточно для качественной абстрактивной суммаризации.
4. **Full FT приводит к переобучению.** 494M параметров на 2000 примерах дают коллапс генерации на тесте (ROUGE близок к 0).
5. **Что улучшать дальше:** больше данных (10K-61K), модель 1.5B-3B, ранняя остановка, дополнительные метрики (например, BERTScore).

## Структура проекта

```text
├── notebooks/
│   ├── 01_eda_baseline.ipynb
│   ├── 02_head_only.ipynb
│   ├── 03_qlora_unsloth.ipynb
│   └── 04_full_ft.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── metrics.py
│   └── baselines.py
├── reports/
│   ├── baseline_rouge.json
│   ├── 02_head_only_rouge.json
│   └── 03_qlora_rouge.json
├── .gitignore
├── requirements.txt
└── README.md
```

## Датасет

**Gazeta Summaries** — 61K русских новостных статей с человеческими резюме.

| Split | Samples |
|-------|---------|
| Train | 60,964 |
| Val | 6,369 |
| Test | 6,793 |

## Модель

**Qwen2.5-0.5B-Instruct** (494M параметров, decoder-only, ChatML)

## Эксперименты

### 1) EDA + Baseline (`01_eda_baseline.ipynb`)
Статистика датасета и три экстрактивных baseline-а, оценка на полном тесте (6,793 примера).

### 2) Head-only (`02_head_only.ipynb`)
Заморозка всех слоев, обучение только `lm_head`. Tied embeddings -> 136M (28%), `float16`, `batch_size=1`, `seq_len=512`.

### 3) QLoRA + Unsloth (`03_qlora_unsloth.ipynb`)
4-bit NF4, LoRA `r=16` (8.8M, 1.7%), `train_on_responses_only`, `batch_size=2`, `seq_len=2048`.

### 4) Full Fine-tuning (`04_full_ft.ipynb`)
494M параметров, gradient checkpointing, `adamw_8bit`, `batch_size=1`, `seq_len=512` -> катастрофическое переобучение.

## Проблемы на Kaggle и фиксы

Ниже перечислены ключевые инженерные проблемы, которые повлияли на стабильность обучения и итоговые метрики.

1. **Поиск датасета (`os.listdir` vs `os.walk`)**  
   Input-датасеты Kaggle лежат в `/kaggle/input/` с непредсказуемой вложенностью. Поиск через `os.listdir` на один уровень не находил `gazeta_train.jsonl`.  
   **Фикс:** рекурсивный поиск через `os.walk`.

2. **`total_mem` vs `total_memory`**  
   Использование `torch.cuda.get_device_properties(0).total_mem` вызывало `AttributeError`.  
   **Фикс:** корректное поле `total_memory`.

3. **`fp16=True` + модель уже в `float16`**  
   При загрузке модели в `torch_dtype=torch.float16` и одновременном `fp16=True` в `TrainingArguments` возникал краш AMP: `Attempting to unscale FP16 gradients`.  
   **Фикс:** `fp16=False`, если модель уже в half precision.

4. **CUDA OOM на T4 (15 GB)**  
   Большой словарь Qwen2.5 (~152K токенов) сильно увеличивает память на кросс-энтропии. Конфиг `batch_size=2`, `seq_len=2048` давал мгновенный OOM.  
   **Фикс:** `batch_size=1`, `seq_len=512`, `gradient_accumulation_steps=16`.

5. **`gradient_checkpointing` с замороженными слоями**  
   В head-only режиме (`requires_grad=False` почти везде) checkpointing приводил к нестабильным CUDA ошибкам.  
   **Фикс:** отключить gradient checkpointing для head-only (для full FT оставлять можно).

6. **Полностью замаскированные labels (`loss=0`)**  
   Маскирование промпта через `[-100] * prompt_len` при несовпадении длины токенизации приводило к тому, что все labels становились `-100`.  
   **Фикс:** для head-only и full FT использовать `labels = input_ids.copy()`, для QLoRA — `train_on_responses_only` из Unsloth.

7. **Monkey-patch Unsloth (`Qwen2Attention`)**  
   После `import unsloth` библиотека патчит `Qwen2Attention`; последующие загрузки стандартного `transformers` в том же процессе могли падать с `apply_qkv`.  
   **Фикс:** разделить эксперименты по разным ноутбукам (отдельные Python-процессы).

8. **`apply_chat_template` возвращает Tensor**  
   В QLoRA-инференсе `tok.apply_chat_template(..., return_tensors="pt")` возвращал `torch.Tensor` без `attention_mask`, из-за чего `generate` падал.  
   **Фикс:** сначала `tokenize=False` (получить текст), затем обычная токенизация `tok(prompt_text, return_tensors="pt")`.

9. **`adamw_8bit` без `bitsandbytes`**  
   `optim="adamw_8bit"` требует установленный `bitsandbytes`; на Kaggle пакет не всегда предустановлен.  
   **Фикс:** явная установка `bitsandbytes` в стартовой ячейке.

10. **`use_reentrant` warning/failures**  
    В новых версиях `transformers` вызов checkpointing без `use_reentrant=False` может приводить к предупреждениям и нестабильному backward.  
    **Фикс:** `model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`.

11. **Checkpointing не отключен перед `generate`**  
    Генерация после обучения при активном gradient checkpointing могла работать некорректно.  
    **Фикс:** `model.gradient_checkpointing_disable()` перед инференсом.

Итог: Kaggle + LLM fine-tuning требует очень аккуратной совместимости версий, точной настройки mixed precision и строгого контроля памяти. Значимая часть деградации метрик связана именно с инфраструктурными и конфигурационными ограничениями среды.

## Запуск

Kaggle -> New Notebook -> Add Input `gazeta-summaries` -> GPU T4, Internet ON -> Upload `.ipynb` -> Run All.

## Технологии

Python 3.12, PyTorch 2.4+, Hugging Face Transformers, TRL, Unsloth, ROUGE-score.
