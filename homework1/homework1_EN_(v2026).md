# Homework Assignment No. 1
*Auto-translated. May contain mistakes.*

## Task
You are required to create and annotate your own image dataset. The dataset must contain **at least 3 classes** with **at least 100 instances** each. Images can be downloaded from the internet or by combining several existing datasets. Train a model on your custom dataset. Create a web application for image classification using the collected dataset. Use data augmentation, regularization, and transfer learning.

## Review Questions
1. Dataset structure, data augmentation.
2. Transfer learning, fine-tuning.
3. Convolutional neural network architecture.

---

## Part 1. Dataset Collection and Preparation

### Image Downloading

There are several ways to obtain images for your dataset. Choose the most convenient option for you:

**Option 1. Manual Collection**

Search for images in search engines (Yandex.Images, Google Images) based on your class topics and download them manually, organizing them into three separate folders.

**Option 2. Using a Browser Extension**

Install one of the browser extensions for batch image downloading, for example:
- [Image downloader - Imageye](https://chrome.google.com/webstore/detail/image-downloader-imageye/agionbommeaifngbhincahgmoflcikhm?hl=en)
- [Image Downloader](https://chrome.google.com/webstore/detail/image-downloader/cnpniohnfphhjihaiiggeabnkjhpaldj?hl=en)

Procedure (example for the selected extension):
1. Perform an image search in a search engine
2. Open the extension
3. Specify the required parameters (file type, size, etc.)
4. Select the desired images and download them

> **Important:** Extension links and interfaces may change over time. Verify the current information at the time of completing the assignment.

![image](images.png)

**Option 3. Using Python Libraries**

There are libraries for programmatic image downloading, for example:
- `yandex-images-download` (for Yandex.Images)
- `google-images-download` (for Google Images)
- Other similar tools

Below is a usage example (verify the currency of commands and parameters in the library documentation):

```bash
# Example for yandex-images-download (requires ChromeDriver)
yandex-images-download Chrome --keywords "cat, dog, bird" --limit 100
```

> **Note:** Libraries and search engine APIs may change. It is recommended to check the currency of the tools immediately before starting the work.

### Image Validation

After downloading the images, **be sure** to perform a manual check:
- Remove duplicates (identical images)
- Remove images that do not match the class theme
- Remove corrupted or invalid files

You are required to keep at least 100 high-quality images per class.

---

## Part 2. Model Training in Google Colab

Model training is performed based on the template and knowledge acquired during the course (Lab Works 1–4).

### Training Requirements

When training the model, you must use:
- **Data Augmentation** — to increase the variability of the training set
- **Regularization** — to prevent overfitting
- **Transfer Learning** — using pre-trained models

### Template for Implementation

For this task, use the prepared Jupyter Notebook:

👉 **[Link to training templates](/TrainModel-notebooks/)**

Workflow:
1. Open the template in Google Colab
2. Create a copy on your Google Drive
3. Upload your collected dataset
4. Implement training according to the requirements
5. Save the trained model in **ONNX** format

### Importing Images into Google Colab

In the Colab file system, in the "content" folder, create three folders corresponding to your image classes (example: "Cake, Swallow, Cat") and upload the previously downloaded images there.

![image](colab.png)

Open the file system in Colab (the "content" folder will open automatically). Right-click on an empty space and create folders for each class.

---

## Part 3. Web Application with FastAPI + Jinja2

In this part, you need to create a web application for image classification using your trained ONNX model.

### Technology Stack

- **FastAPI** — a web framework for building APIs and web applications
- **Jinja2** — a templating engine for generating HTML pages
- **ONNX Runtime** — for running model inference
- **Uvicorn** — an ASGI server for running the application

Detailed instructions for creating the web application are provided in a separate document:

👉 **[Link to FastAPI + Jinja2 application guide](/API-Jinja/FastApi-Jinja-instruct.md)**

### Main Steps:

1. **Installing libraries:**
```bash
pip install fastapi==0.115.0 starlette==0.41.0 jinja2==3.1.4 python-multipart onnxruntime pillow
```

2. **Creating the project structure:**
```
project/
├── main.py                 # Main application file
├── templates/
│   └── index.html          # HTML template
├── media/
│   ├── images/             # Uploaded images
│   └── models/             # ONNX model
│       └── model.onnx
```

3. **Implementing the application:**
   - Loading the ONNX model
   - Image preprocessing
   - Running inference
   - Displaying the result

4. **Running the application:**
```bash
uvicorn main:app --reload
```

### Launch and Testing

After starting the application:
- Open your browser at `http://127.0.0.1:8000`
- Upload a test image
- Verify the classification result

---

## Links and Materials

| Material | Description |
|----------|-------------|
| [Colab training template](/TrainModel-notebooks/) | Jupyter Notebook for model training |
| [FastAPI + Jinja2 guide](/API-Jinja/FastApi-Jinja-instruct.md) | Detailed guide for building the web application |
| [ONNX Runtime documentation](https://onnxruntime.ai/) | Working with ONNX models |
| [FastAPI documentation](https://fastapi.tiangolo.com/) | Official FastAPI documentation |