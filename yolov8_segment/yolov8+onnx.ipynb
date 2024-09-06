{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "# **YOLOv8.2-segment**\n",
        "\n",
        "Это руководство основано на [репозитории ultralytics](ultralytics).\n",
        "  \n",
        "Тип разметки для YOLO - набор точек объектов\n",
        "# **YOLOv8.2-segment**\n",
        "\n",
        "本指南基于[ultralytics](ultralytics)。\n",
        "\n",
        "YOLO 的标记类型 - 一组轮廓点"
      ],
      "metadata": {
        "id": "GwqkDg11f31b"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **1. Установка зависимостей**\n",
        "# **1. 安装依赖项**"
      ],
      "metadata": {
        "id": "XtpCH2iBiTFY"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 1.1 Монтирование Google Drive  \n",
        "Можно не монтировать гугл диск, а хранить все данные для выполнения дз в файлах ноутбука, но тогда репозиторий YOLO, результаты обучения и датасет не сохранятся при отключении от среды выполнения. То есть каждый раз надо будет клонировать репозиторий и загружать датасет, что не очень удобно.\n",
        "# 1.1 加载Google Drive\n",
        "您可以不挂载Google驱动器，而是将执行任务的所有数据存储在笔记本电脑文件中，但是当与运行时环境断开连接时，YOLO存储库、训练结果和数据集将不会被保存。也就是说，每次都需要克隆存储库并下载数据集，这不是很方便。"
      ],
      "metadata": {
        "id": "Eefpe6XBiUGR"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BhQgxpYvfebg",
        "outputId": "797776de-a6d2-45be-a197-8872c37be2fe"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount(\"/content/gdrive\", force_remount=True).\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/gdrive')"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Определяем видеокарту GPU, чтобы на ней учить нейронную сеть\n",
        "## 确定 GPU 显卡以便在其上训练神经网络"
      ],
      "metadata": {
        "id": "jZeeL41Xip44"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!nvidia-smi"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Htkta6FplhoB",
        "outputId": "ca76ca22-07ac-4aee-a073-818febbedacc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/bin/bash: line 1: nvidia-smi: command not found\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 1.2 Установка зависимостей\n",
        "# 1.2 安装依赖"
      ],
      "metadata": {
        "id": "v4lqWXKUi5Wg"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "%pip install ultralytics\n",
        "import ultralytics\n",
        "import os\n",
        "import json\n",
        "import shutil\n",
        "import yaml\n",
        "from sklearn.model_selection import train_test_split\n",
        "ultralytics.checks()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3jPRUP9WfkE1",
        "outputId": "ef4f3225-0eb5-40c6-9a1e-96d5a267c531"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Ultralytics YOLOv8.2.34 🚀 Python-3.10.12 torch-2.3.0+cu121 CPU (Intel Xeon 2.20GHz)\n",
            "Setup complete ✅ (2 CPUs, 12.7 GB RAM, 30.3/107.7 GB disk)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **2. Импорт датасета**\n",
        "#**2.导入数据集**"
      ],
      "metadata": {
        "id": "ZiaoMZ-_jQi4"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Набор данных должен быть экспортирован из cvat в формате COCO 1.0.  \n",
        "\n",
        "В бесплатной онлайн-версии нельзя экспортировать датасет вместе с картинками, если используется эта версия, картинки надо загружать отдельно. Если устанавливать cvat локально, такого ограничения нет.\n",
        "\n",
        "Загрузите аннотации и картинки в одну папку и загрузите ее на google диск,туда же поместите файл instances_default.json.\n",
        "\n",
        "数据集必须以 COCO 1.0 格式从 cvat 导出。\n",
        "\n",
        "在免费在线版本中，您无法将数据集与图片一起导出；如果您使用此版本，则必须单独上传图片。如果本地安装cvat则没有这样的限制。\n",
        "\n",
        "将注释和图片上传到一个文件夹中，然后将其上传到Google Drive，将instances_default.json 文件也放在那里。"
      ],
      "metadata": {
        "id": "1nhMlXJEvs6R"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def convert_coco_to_yolo_segmentation(json_file, folder_name = \"labels\"):\n",
        "    folder_name = folder_name\n",
        "    # Load the JSON file\n",
        "    with open(json_file, 'r') as file:\n",
        "        coco_data = json.load(file)\n",
        "\n",
        "    # Create a \"labels\" folder to store YOLO segmentation annotations\n",
        "    output_folder = os.path.join(os.path.dirname(json_file), folder_name)\n",
        "    os.makedirs(output_folder, exist_ok=True)\n",
        "\n",
        "    # Extract annotations from the COCO JSON data\n",
        "    annotations = coco_data['annotations']\n",
        "    for annotation in annotations:\n",
        "        image_id = annotation['image_id']\n",
        "        category_id = annotation['category_id']\n",
        "        segmentation = annotation['segmentation']\n",
        "        bbox = annotation['bbox']\n",
        "\n",
        "        # Find the image filename from the COCO data\n",
        "        for image in coco_data['images']:\n",
        "            if image['id'] == image_id:\n",
        "                image_filename = os.path.basename(image['file_name'])\n",
        "                image_filename = os.path.splitext(image_filename)[0] # Removing the extension. (In our case, it is the .jpg or .png part.)\n",
        "                image_width = image['width']\n",
        "                image_height = image['height']\n",
        "                break\n",
        "\n",
        "        # Calculate the normalized center coordinates and width/height\n",
        "        x_center = (bbox[0] + bbox[2] / 2) / image_width\n",
        "        y_center = (bbox[1] + bbox[3] / 2) / image_height\n",
        "        bbox_width = bbox[2] / image_width\n",
        "        bbox_height = bbox[3] / image_height\n",
        "\n",
        "        # Convert COCO segmentation to YOLO segmentation format\n",
        "        yolo_segmentation = [f\"{(x) / image_width:.5f} {(y) / image_height:.5f}\" for x, y in zip(segmentation[0][::2], segmentation[0][1::2])]\n",
        "        #yolo_segmentation.append(f\"{(segmentation[0][0]) / image_width:.5f} {(segmentation[0][1]) / image_height:.5f}\")\n",
        "        yolo_segmentation = ' '.join(yolo_segmentation)\n",
        "\n",
        "        # Generate the YOLO segmentation annotation line\n",
        "        yolo_annotation = f\"{category_id} {yolo_segmentation}\"\n",
        "\n",
        "        # Save the YOLO segmentation annotation in a file\n",
        "        output_filename = os.path.join(output_folder, f\"{image_filename}.txt\")\n",
        "        with open(output_filename, 'a+') as file:\n",
        "            file.write(yolo_annotation + '\\n')\n",
        "\n",
        "    print(\"Conversion completed. YOLO segmentation annotations saved in 'labels' folder.\")\n",
        "\n",
        "# Example usage\n",
        "json_file = \"/content/gdrive/MyDrive/dataset/instances_default.json\" #JSON file\n",
        "split = \"/content/gdrive/MyDrive/dataset\" #Folder\n",
        "convert_coco_to_yolo_segmentation(json_file, split)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JU_l8qPKuxdR",
        "outputId": "307e5dee-25e6-42e7-b611-9661da76c017"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Conversion completed. YOLO segmentation annotations saved in 'labels' folder.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "dataset_path = '/content/gdrive/MyDrive/dataset' # Укажите свой путь до датасета"
      ],
      "metadata": {
        "id": "mtIFC0FznX17"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test_size = 0.2 # размер тестовой выборки\n",
        "valid_size = 0.1 # размер валидационной выборки"
      ],
      "metadata": {
        "id": "7B4Sstg8nYg8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]\n",
        "annotation_files = [f for f in os.listdir(dataset_path) if f.endswith('.txt')]\n",
        "\n",
        "image_files_with_annotations = [f for f in image_files if f.replace('.jpg', '.txt') in annotation_files]\n",
        "\n",
        "train_files, test_files = train_test_split(image_files_with_annotations, test_size=test_size, random_state=42)\n",
        "train_files, valid_files = train_test_split(train_files, test_size=len(image_files_with_annotations)*valid_size/len(train_files), random_state=42)\n",
        "\n",
        "def copy_files(files, source_path, dest_path, folder):\n",
        "    for file in files:\n",
        "        shutil.copy(os.path.join(source_path, file), os.path.join(dest_path, 'images', folder))\n",
        "        shutil.copy(os.path.join(source_path, file.replace('.jpg', '.txt')), os.path.join(dest_path, 'labels', folder))\n",
        "\n",
        "for folder in ['train', 'val', 'test']:\n",
        "    os.makedirs(os.path.join(dataset_path, 'images', folder), exist_ok=True)\n",
        "    os.makedirs(os.path.join(dataset_path, 'labels', folder), exist_ok=True)\n",
        "\n",
        "copy_files(train_files, dataset_path, dataset_path, 'train')\n",
        "copy_files(valid_files, dataset_path, dataset_path, 'val')\n",
        "copy_files(test_files, dataset_path, dataset_path, 'test')\n",
        "\n",
        "for file in os.listdir(dataset_path):\n",
        "    if not file.endswith('.names') and os.path.isfile(os.path.join(dataset_path, file)):\n",
        "      os.remove(os.path.join(dataset_path, file))"
      ],
      "metadata": {
        "id": "Q838yqHImgTP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_path = f'{dataset_path}/images/train'\n",
        "val_path = f'{dataset_path}/images/val'\n",
        "test_path = f'{dataset_path}/images/test'\n",
        "\n",
        "# names_file = os.path.join(dataset_path, 'obj.names')\n",
        "# with open(names_file, 'r') as file:\n",
        "#     class_names = [line.strip() for line in file.readlines()]\n",
        "\n",
        "# num_classes = len(class_names)\n",
        "\n",
        "data = {\n",
        "    'path': dataset_path,\n",
        "    'train': train_path,\n",
        "    'val': val_path,\n",
        "    'test': test_path,\n",
        "    'names': {\n",
        "      0: 'airplane',\n",
        "      1: 'weather_balloon',\n",
        "      2: 'UFO'\n",
        "    }\n",
        "}\n",
        "\n",
        "output_file = os.path.join(dataset_path, 'data.yaml')\n",
        "with open(output_file, 'w') as file:\n",
        "    documents = yaml.dump(data, file)\n",
        "\n",
        "# os.remove(os.path.join(dataset_path, 'obj.names'))"
      ],
      "metadata": {
        "id": "mxXuajqBp1JO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **3. Запуск обучения**\n",
        "#**3.开始训练**"
      ],
      "metadata": {
        "id": "N-FYqX-ajk39"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "%cd /content/gdrive/MyDrive"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tvcGzIkVfuNO",
        "outputId": "20ab2dcf-a894-464c-f035-c412dfa35dee"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/content/gdrive/MyDrive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 3.1 Начало обучения\n",
        "\n",
        "[Чтобы получить полный список аргументов обучения, перейдите по ссылке](https://docs.ultralytics.com/modes/train/#train-settings)\n",
        "\n",
        "#3.1 训练开始\n",
        "\n",
        "[有关训练参数的完整列表，请点击链接](https://docs.ultralytics.com/modes/train/#train-settings)"
      ],
      "metadata": {
        "id": "dAuT4Chuj4Ou"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Load YOLOv8n-seg, train it on COCO128-seg for 3 epochs and predict an image with it\n",
        "from ultralytics import YOLO\n",
        "\n",
        "model = YOLO('yolov8n-seg.pt')  # load a pretrained YOLOv8n segmentation model\n",
        "model.train(data='dataset/data.yaml', epochs=10, imgsz=640, device=0)  # train the model\n",
        "model('datasets_seg/images/train/airplane(0).jpeg')  # predict on an image"
      ],
      "metadata": {
        "id": "cL6YebNIflvH"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **4. Экспорт модели в onnx**\n",
        "# **4.将模型导出到 onnx**"
      ],
      "metadata": {
        "id": "E7dVz2pTrg_P"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install ultralytics onnx onnx-simplifier onnxruntime"
      ],
      "metadata": {
        "id": "ZN1UkWYIf9c9"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from ultralytics import YOLO\n",
        "\n",
        "# Load a YOLO-seg model\n",
        "model = YOLO('runs/segment/train/weights/best.pt')\n",
        "\n",
        "# Export to ONNX\n",
        "onnx_file = model.export(format='onnx')\n",
        "\n",
        "# Load an ONNX model\n",
        "onnx_model = YOLO(onnx_file)\n",
        "\n",
        "# Run inference on CPU and GPU with ONNXRuntime\n",
        "results = onnx_model('dataset/images/test/airplane(19).jpg')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tsDtzcFh_qBi",
        "outputId": "b5f29e96-d0ec-43ff-bb2d-21a9ba06b364"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Ultralytics YOLOv8.2.27 🚀 Python-3.10.12 torch-2.3.0+cu121 CPU (Intel Xeon 2.20GHz)\n",
            "YOLOv8n-seg summary (fused): 195 layers, 3258649 parameters, 0 gradients, 12.0 GFLOPs\n",
            "\n",
            "\u001b[34m\u001b[1mPyTorch:\u001b[0m starting from 'runs/segment/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) ((1, 39, 8400), (1, 32, 160, 160)) (6.5 MB)\n",
            "\n",
            "\u001b[34m\u001b[1mONNX:\u001b[0m starting export with onnx 1.16.1 opset 17...\n",
            "\u001b[34m\u001b[1mONNX:\u001b[0m export success ✅ 1.5s, saved as 'runs/segment/train/weights/best.onnx' (12.6 MB)\n",
            "\n",
            "Export complete (3.6s)\n",
            "Results saved to \u001b[1m/content/gdrive/MyDrive/runs/segment/train/weights\u001b[0m\n",
            "Predict:         yolo predict task=segment model=runs/segment/train/weights/best.onnx imgsz=640  \n",
            "Validate:        yolo val task=segment model=runs/segment/train/weights/best.onnx imgsz=640 data=datasets_seg/data.yaml  \n",
            "Visualize:       https://netron.app\n",
            "Loading runs/segment/train/weights/best.onnx for ONNX Runtime inference...\n",
            "\n",
            "image 1/1 /content/gdrive/MyDrive/dataset/test/images/airplane(11).jpg: 640x640 1 airplane, 397.4ms\n",
            "Speed: 6.0ms preprocess, 397.4ms inference, 8.5ms postprocess per image at shape (1, 3, 640, 640)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(results)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "UaKv1xk0APog",
        "outputId": "f89b1659-1187-4c0a-a134-6695f4acb890"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[ultralytics.engine.results.Results object with attributes:\n",
            "\n",
            "boxes: ultralytics.engine.results.Boxes object\n",
            "keypoints: None\n",
            "masks: ultralytics.engine.results.Masks object\n",
            "names: {0: 'airplane', 1: 'weather_balloon', 2: 'UFO'}\n",
            "obb: None\n",
            "orig_img: array([[[184, 156, 132],\n",
            "        [184, 156, 132],\n",
            "        [185, 154, 131],\n",
            "        ...,\n",
            "        [159, 117,  88],\n",
            "        [159, 117,  88],\n",
            "        [159, 117,  88]],\n",
            "\n",
            "       [[183, 155, 131],\n",
            "        [183, 155, 131],\n",
            "        [184, 153, 130],\n",
            "        ...,\n",
            "        [159, 117,  88],\n",
            "        [159, 117,  88],\n",
            "        [159, 117,  88]],\n",
            "\n",
            "       [[182, 154, 130],\n",
            "        [182, 154, 130],\n",
            "        [183, 152, 129],\n",
            "        ...,\n",
            "        [158, 116,  87],\n",
            "        [158, 116,  87],\n",
            "        [158, 116,  87]],\n",
            "\n",
            "       ...,\n",
            "\n",
            "       [[165, 156, 152],\n",
            "        [165, 156, 152],\n",
            "        [165, 156, 152],\n",
            "        ...,\n",
            "        [171, 170, 174],\n",
            "        [171, 169, 175],\n",
            "        [172, 170, 176]],\n",
            "\n",
            "       [[165, 156, 152],\n",
            "        [166, 157, 153],\n",
            "        [164, 158, 153],\n",
            "        ...,\n",
            "        [172, 171, 175],\n",
            "        [172, 170, 176],\n",
            "        [172, 170, 176]],\n",
            "\n",
            "       [[165, 156, 152],\n",
            "        [166, 157, 153],\n",
            "        [164, 158, 153],\n",
            "        ...,\n",
            "        [172, 171, 175],\n",
            "        [172, 170, 176],\n",
            "        [172, 170, 176]]], dtype=uint8)\n",
            "orig_shape: (1706, 2560)\n",
            "path: '/content/gdrive/MyDrive/dataset/test/images/airplane(11).jpg'\n",
            "probs: None\n",
            "save_dir: 'runs/segment/predict'\n",
            "speed: {'preprocess': 5.961179733276367, 'inference': 397.43638038635254, 'postprocess': 8.535385131835938}]\n"
          ]
        }
      ]
    }
  ]
}