# Parts on conveyor

**Название команды:** Artyom Prokopets

**Участники:**
- Прокопец Артем Игоревич, группа 23КНТ-6, Kaggle: artyomprokopets

**ID финальной посылки Kaggle:** 58747567

## Параметры финальной модели

- Архитектура: ResNet18 с предобученными весами ImageNet.
- Версия датасета: v2.
- Seed: 42.
- Количество эпох обучения: 40.
- Выбранная эпоха (лучший checkpoint): 32.
- Размер входного изображения: 128 × 128.
- Batch size: 64.
- Количество параметров: 11 180 616.
- Критерий выбора checkpoint: максимальный Macro-F1 на validation.
- Validation Macro-F1: 0.995832.
- Validation Accuracy: 0.995833.

## Команды запуска

**Обучение:**

```bash
python train.py --data . --out runs/exp10_resize128_40 --epochs 40 --seed 42
```

**Предсказание:**

```bash
python predict.py --data . --weights model.pt --output submission_reproduced.csv
```
