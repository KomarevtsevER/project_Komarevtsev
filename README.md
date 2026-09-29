#  Mobile Price Classification — EDA

Exploratory Data Analysis (EDA) датасета **Mobile Price Classification** с Kaggle.  
Цель проекта — исследовать зависимости между техническими характеристиками мобильных телефонов и их ценовой категорией.

## Установка и запуск

### 1. Клонирование репозитория

```bash
git clone <URL_РЕПОЗИТОРИЯ>
cd project_Komarevtsev
```

## Создание виртуального окружения

```bash
python -m venv .venv_project_Komarevtsev
source .venv_project_Komarevtsev/bin/activate  # Linux/Mac
# или
.venv_project_Komarevtsev\Scripts\activate     # Windows
```
## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Размещение данных

Поместите файл dataset.csv в папку data/.
Датасет можно скачать с Kaggle:
 https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification

## Описание датасета

Датасет содержит 2000 записей о мобильных телефонах с 21 признаком:
Категориальные признаки

blue, dual_sim, four_g, three_g, touch_screen, wifi — бинарные (0/1)
n_cores — количество ядер процессора (1-8)
Числовые признаки

battery_power — ёмкость батареи (mAh)
clock_speed — тактовая частота процессора (GHz)
fc — мегапиксели фронтальной камеры
int_memory — внутренняя память (GB)
m_dep — толщина телефона (cm)
mobile_wt — вес телефона (g)
pc — мегапиксели основной камеры
px_height, px_width — разрешение экрана (px)
ram — оперативная память (MB)
sc_h, sc_w — размеры экрана (cm)
talk_time — время разговора (часы)
Целевая переменная

price_range — ценовая категория: 0 (низкая), 1 (средняя), 2 (высокая), 3 (очень высокая)
Проведённый анализ

## Очистка данных

Удалены аномальные значения (sc_w < 5)
Оптимизированы типы данных (int8, int16, float32) для экономии памяти

## Feature Engineering

Созданы новые признаки:
total_pixels — общее разрешение экрана
battery_efficiency — эффективность батареи
total_camera — суммарная мощность камер

## Ключевые выводы

RAM — наиболее значимый признак (корреляция с ценой ~0.92)
Дорогие телефоны имеют больше RAM, ёмкую батарею, высокое разрешение экрана и мощные камеры
Наличие 4G, Wi-Fi и сенсорного экрана сильно связано с высокой ценой
Обнаружена мультиколлинеарность между px_height/px_width и sc_h/sc_w

# MLFlow

### Расположение скрипта
project_Komarevtsev/
└── mlflow/
    └── start_mlflow.sh    ← скрипт запуска

### Команда запуска 

1. Перейдите в папку mlflow
cd mlflow

2. Запустите скрипт
sh start_mlflow.sh

После запуска откройте браузер по адресу: http://127.0.0.1:5000

# Результаты обучение моделей 
### Лучшая модель 

В ходе исследования были проведены эксперименты с различными подходами к обработке признаков и моделям машинного обучения. Наилучшее качество показала модель CatBoostClassifier с экспертным отбором признаков, обученная на полной выборке.

CatBoost metrics:
   precision: 0.9345
   recall: 0.9343
   f1: 0.9343
   roc_auc: 0.9955


| Параметр | Значение |
|---|---|
| Алгоритм | `CatBoostClassifier` |
| iterations | `200` |
| depth | `6` |
| learning_rate | `0.1` |
| random_seed | `42` |
| catboost_version | `1.2.10` |
| mlflow_version | `2.16.0` |
| Размер модели | `543 199 байт (~530 КБ)` |

| № | Признак | Тип |
|---|---------|-----|
| 1 | `battery_power` | integer |
| 2 | `blue` | long |
| 3 | `clock_speed` | float |
| 4 | `dual_sim` | long |
| 5 | `fc` | integer |
| 6 | `four_g` | long |
| 7 | `int_memory` | integer |
| 8 | `m_dep` | float |
| 9 | `mobile_wt` | integer |
| 10 | `n_cores` | long |
| 11 | `pc` | integer |
| 12 | `px_height` | integer |
| 13 | `px_width` | integer |
| 14 | `ram` | integer |
| 15 | `sc_h` | integer |
| 16 | `sc_w` | integer |
| 17 | `talk_time` | integer |
| 18 | `three_g` | long |
| 19 | `touch_screen` | long |
| 20 | `wifi` | long |
| 21 | `total_pixels` | integer |
| 22 | `pixel_density` | double |
| 23 | `battery_efficiency` | float |
| 24 | `total_camera` | integer |



## run_id: 339fa3e1a0bb41d881b37f37d3aeb33a