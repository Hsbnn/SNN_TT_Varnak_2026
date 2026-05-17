# SNN + Tensor-Train research package

Скрипты реализуют три блока исследований из ВКР
«Применение тензорных поездов для ускорения работы импульсных нейросетей».

## Структура

```
snn_tt_research/
├── common/                       общие утилиты для всех трёх блоков
│   ├── device.py                 выбор устройства, синхронизация, seed
│   ├── spike.py                  Heaviside + surrogate gradient, Poisson
│   ├── encoding.py               Gaussian RF для табличных и 2-D входов
│   ├── tt_decomp.py              TT-SVD, реконструкция, low-rank SVD
│   ├── metrics.py                подсчёт параметров и MAC-ов
│   ├── benchmarks.py             медианный бенчмарк, фиксированные spike-batch
│   ├── evaluation.py             precision / recall / F1 / kappa / top-k
│   └── plots.py                  графики: матрица ошибок, per-class bar, сравнения
├── block1_iris/                  Iris + RF + ANN/SNN/SNN+TT + STDP
├── block2_mnist_stdp/            MNIST + STDP-кодировщик + TT-сжатие + metrics_report
├── block3_fashion_surgrad/       Fashion-MNIST + surrogate gradient + TT + LowRank + metrics_report
└── main.py                       CLI: запуск нужного блока
```

Все три блока самостоятельны и могут запускаться по отдельности; общий код
вынесен в `common` и не дублируется между блоками.

## Запуск

```
python -m snn_tt_research.main block1
python -m snn_tt_research.main block2 --data-root data --plots-dir plots
python -m snn_tt_research.main block3 --data-root data --plots-dir plots
python -m snn_tt_research.main all
```

Каждый блок возвращает датакласс с числовыми результатами: точности, параметры,
MAC-и, латентности, а для блоков 2 и 3 — словарь `ClassificationReport` по каждой
модели (accuracy, macro precision/recall/F1, weighted-варианты, balanced accuracy,
Cohen's kappa, top-2 / top-5 accuracy, per-class precision/recall/F1 и матрица
ошибок) плюс пути ко всем сохранённым `.png`-графикам.

## Графики

В директории `plots/<блок>/` автоматически сохраняются:

- матрица ошибок для каждой модели (`confusion_<name>.png`),
- столбчатая диаграмма precision/recall/F1 по классам (`per_class_<name>.png`),
- сравнение моделей по accuracy/precision/recall/F1 (`models_comparison.png`),
- кривые валидационной точности по эпохам (`training_curves.png`, блок 3),
- accuracy vs число параметров (`accuracy_vs_params.png`),
- латентность + ускорение относительно dense (`latency_speedup.png`).

## Зависимости

`torch`, `numpy`, `pandas`, `scikit-learn`, `torchvision`, `tqdm`, `matplotlib`.
