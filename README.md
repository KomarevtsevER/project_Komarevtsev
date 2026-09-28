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