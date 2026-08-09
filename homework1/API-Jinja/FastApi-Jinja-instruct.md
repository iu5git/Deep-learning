# Создание веб-приложения на FastAPI + Jinja2

# Введение

## Что такое FastAPI?

**FastAPI** — это современный веб-фреймворк для Python, предназначенный для создания веб-приложений и API. Он основан на стандартах **OpenAPI** и **JSON Schema**, что обеспечивает автоматическую генерацию документации.

**Ключевые особенности:**
- **Высокая производительность** — один из самых быстрых Python-фреймворков (на уровне Node.js и Go)
- **Асинхронность** — поддержка `async/await` для эффективной обработки запросов
- **Автоматическая документация** — Swagger UI и ReDoc генерируются автоматически по адресам `/docs` и `/redoc`
- **Type Hints** — использование аннотаций типов Python делает код самодокументируемым
- **Встроенная валидация** — через Pydantic модели данных
- **Простота** — минимальный код для создания работающего приложения

## Что такое Jinja2?

**Jinja2** — это шаблонизатор для Python, вдохновленный шаблонами Django. Он позволяет создавать HTML-страницы с динамическим содержимым, встраивая Python-код в HTML-шаблоны.

**Основные возможности:**
- Вставка переменных: `{{ variable }}`
- Условные конструкции: `{% if condition %} ... {% endif %}`
- Циклы: `{% for item in items %} ... {% endfor %}`
- Наследование шаблонов: `{% extends "base.html" %}`
- Фильтры: `{{ value|upper }}`, `{{ value|length }}`

> **💡 Аналогия:** Jinja2 концептуально очень похож на **Razor** в .NET (C#) или **Blade** в Laravel (PHP). Если вы знакомы с этими технологиями, синтаксис Jinja2 покажется вам интуитивно понятным — он также использует специальные теги для вставки кода в HTML-разметку.

## Сравнение с Django

| Характеристика | Django | FastAPI |
|----------------|--------|---------|
| **Архитектура** | Монолитная, "батарейки включены" | Легковесная, модульная |
| **Скорость** | Медленнее (синхронный) | Быстрее (асинхронный) |
| **ORM** | Встроенная | Отсутствует (можно добавить SQLAlchemy) |
| **Админ-панель** | Встроенная | Отсутствует |
| **Шаблонизатор** | Собственный (Django Templates) | Любой (рекомендуется Jinja2) |
| **Сложность** | Высокая | Низкая |
| **Документация API** | Требует ручной настройки | Автоматическая (OpenAPI) |
| **Использование** | Крупные веб-приложения | API, микросервисы, прототипы |

**Для задачи классификации изображений FastAPI идеально подходит** — нам не нужна админ-панель, ORM и другие "тяжелые" компоненты Django. FastAPI позволяет быстро создать легкое веб-приложение с минимальным кодом.

---

# Реализация

## Шаг 1: Установка библиотек

> **Важно:** Для стабильной работы на Windows рекомендуется использовать проверенную комбинацию версий. Поздние версии FastAPI и Starlette могут иметь проблемы совместимости с Jinja2.

Установите библиотеки:

```bash
pip install fastapi==0.115.0 starlette==0.41.0 jinja2==3.1.4 uvicorn
```

```bash
pip install python-multipart onnxruntime pillow
```

**Назначение каждой библиотеки:**
- `fastapi` — основной веб-фреймворк
- `uvicorn` — ASGI-сервер для запуска приложения
- `jinja2` — шаблонизатор для HTML-страниц
- `python-multipart` — для обработки загружаемых файлов
- `onnxruntime` — для запуска ONNX-моделей
- `pillow` — для обработки изображений

---

## Шаг 2: Создание проекта

Создайте следующую структуру папок и файлов в корне вашего проекта:

```
<project_name>/             # Папка проекта (название по вашему выбору)
├── main.py                 # Основной файл приложения (все маршруты и логика)
├── templates/              # HTML-шаблоны
│   └── index.html          # Главная страница
├── media/                  # Папка для пользовательских файлов
│   ├── images/             # Загруженные изображения
│   └── models/             # ONNX-модели
│       └── <model>.onnx    # Ваша обученная модель
```

---

## Шаг 3: Код приложения

### 3.1. Создайте файл `main.py` — главный файл приложения

**Основные компоненты файла `main.py`:**

1. **Импорт библиотек:**
   - FastAPI, Request, File, UploadFile
   - Jinja2Templates для работы с шаблонами
   - StaticFiles для раздачи статики (медиа-файлов)
   - ONNX Runtime для инференса модели
   - PIL (Pillow) для обработки изображений
   - NumPy для работы с массивами

2. **Инициализация приложения:**
   - Создание экземпляра FastAPI
   - Настройка Jinja2Templates с указанием папки `templates`
   - Монтирование папки `media` как статической для доступа к загруженным файлам

3. **Загрузка ONNX-модели:**
   - Загрузка модели из папки `media/models/`

4. **Функция предобработки изображения:**
   - Открытие и ресайз изображения
   - Нормализация (параметры должны совпадать с обучением)
   - Преобразование в формат, ожидаемый моделью (NCHW или NHWC)

5. **Функция инференса:**
   - Вызов ONNX-модели
   - Softmax для получения вероятностей
   - Возврат предсказанного класса и уверенности

6. **Маршруты:**
   - `GET /` — главная страница с формой загрузки
   - `POST /predict` — обработка загруженного файла и возврат результата

```python
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import onnxruntime as ort
import numpy as np
from PIL import Image

app = FastAPI(title="Классификатор изображений")

# Настройка шаблонов и статики
templates = Jinja2Templates(directory="templates")
app.mount("/media", StaticFiles(directory="media"), name="media")

# Список классов - ИЗМЕНИТЕ НА СВОИ!
CLASS_NAMES = ['Девочка', 'Выдра', 'Фургон']

# Загрузка модели ONNX, вставьте свою
model_path = Path("media/models/<model>.onnx")
session = ort.InferenceSession(str(model_path))

def preprocess_image(image_path: Path):
    """Подготовка изображения для инференса (NCHW)"""
    # Открываем и ресайзим
    img = Image.open(image_path).convert("RGB")
    img = img.resize((32, 32))
    img_array = np.asarray(img).astype(np.float32)
    
    # HWC -> CHW / (32,32,3) -> (3,32,32)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Добавляем batch dimension: (1, 3, 32, 32)
    return np.expand_dims(img_array, axis=0)

def predict_image(image_path: Path):
    """Выполнение инференса"""
    input_batch = preprocess_image(image_path)
    outputs = session.run(None, {'input': input_batch})
    
    # Softmax для вероятностей
    exp_scores = np.exp(outputs[0])
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    predicted_class_id = np.argmax(probabilities[0])
    confidence = probabilities[0][predicted_class_id]
    
    return CLASS_NAMES[predicted_class_id], confidence

# API хэндлеры

@app.get("/")
async def index(request: Request):
    """Главная страница с формой загрузки"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """Обработка загруженного изображения и предсказание"""
    # Сохраняем файл
    upload_dir = Path("media/images")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Делаем предсказание
    predicted_class, confidence = predict_image(file_path)
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": predicted_class,
            "confidence": f"{confidence:.2%}",
            "image_path": f"/media/images/{file.filename}"
        }
    )
```

---

### 3.2. Создайте файл `templates/index.html` — HTML-шаблон

**Основные компоненты шаблона:**

1. **Форма загрузки:**
   - `action="/predict"` — отправка на эндпоинт предсказания
   - `method="post"` — метод POST
   - `enctype="multipart/form-data"` — для загрузки файлов
   - Поле `input type="file" name="file"` — выбор изображения

2. **Отображение результата:**
   - Использование Jinja2-синтаксиса `{% if prediction %}`
   - Вывод предсказанного класса: `{{ prediction }}`
   - Вывод уверенности: `{{ confidence }}`
   - Отображение загруженного изображения: `{{ image_path }}`

```html
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Классификатор изображений</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .upload-form {
            border: 2px dashed #bdc3c7;
            padding: 40px;
            text-align: center;
            border-radius: 10px;
            margin: 20px 0;
            background: #f8f9fa;
        }
        .upload-form input[type="file"] {
            margin: 10px 0;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        button {
            background: #3498db;
            color: white;
            padding: 12px 40px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background: #2980b9;
        }
        .result {
            background: #f0f8ff;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
            border: 1px solid #b0d4f1;
        }
        .prediction {
            font-size: 32px;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0;
        }
        .confidence {
            color: #7f8c8d;
            font-size: 18px;
        }
        .uploaded-image {
            max-width: 300px;
            border-radius: 10px;
            margin: 15px auto;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .footer {
            text-align: center;
            color: #95a5a6;
            margin-top: 30px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>Классификатор изображений</h1>
    <p style="text-align: center; color: #7f8c8d;">
        Загрузите изображение для классификации
    </p>
    
    <div class="upload-form">
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <br><br>
            <button type="submit">Классифицировать</button>
        </form>
    </div>
    
    {% if prediction %}
    <div class="result">
        <h2>Результат классификации:</h2>
        <div class="prediction">{{ prediction }}</div>
        <div class="confidence">Уверенность: {{ confidence }}</div>
        {% if image_path %}
        <img src="{{ image_path }}" alt="Ваше изображение" class="uploaded-image">
        {% endif %}
    </div>
    {% endif %}
    
    <div class="footer">
        Домашнее задание №1 по дисциплине "Разработка нейросетевых систем"
    </div>
</body>
</html>
```

---

## Шаг 4: Настройка параметров под вашу модель

В файле `main.py` необходимо изменить следующие параметры:

### 4.1. Список классов

```python
CLASS_NAMES = ['Торт', 'Ласточка', 'Кошка']  # Замените на свои классы
```

### 4.2. Имя входного слоя

Название входного слоя в ONNX-модели может отличаться от `'input'`. Проверить можно с помощью:

```python
print([inp.name for inp in session.get_inputs()])
```

Исправьте в коде:
```python
outputs = session.run(None, {'input': input_batch})  # 'input' — имя слоя
```

### 4.3. Путь к модели

```python
model_path = Path("media/models/cifar100.onnx")  # Укажите путь к вашей модели
```

---

## Шаг 5: Запуск приложения

### 5.1. Запуск через Uvicorn

Из корневой папки проекта выполните:

```bash
uvicorn main:app --reload
```

**Параметры:**
- `main:app` — файл `main.py`, переменная `app`
- `--reload` — автоматическая перезагрузка при изменении кода (только для разработки)

### 5.3. Проверка работы

После запуска:
- Приложение доступно: `http://127.0.0.1:8000`
- Автоматическая документация API: `http://127.0.0.1:8000/docs`