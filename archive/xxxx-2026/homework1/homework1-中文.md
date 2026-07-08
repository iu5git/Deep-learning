# 作业 №1

## 任务

需要创建并标注一个包含图像的数据集。该数据集应包含不少于3个类别，每个类别不少于100个样本。图像可以从互联网上下载或结合多个现有数据集。创建一个用于对该数据集进行图像分类的Web应用程序。使用数据增强、正则化和迁移学习。

## <mark>问题小测</mark>

1. 数据集的结构和数据增强。
2. 迁移学习和微调。
3. 卷积神经网络的架构。

## 第1部分. 在Google Colab中<mark>训练模型</mark>并保存

### 图像下载

+ **方案 1**. 需要在某个搜索引擎中查找所需类别的图片，然后手动下载到3个不同的文件夹中。
+ **方案 2**. 需要安装Chrome浏览器的扩展程序：[Image downloader - Imageye](https://chrome.google.com/webstore/detail/image-downloader-imageye/agionbommeaifngbhincahgmoflcikhm?hl=en-US//) 或 [Image Downloader](https://chrome.google.com/webstore/detail/image-downloader/cnpniohnfphhjihaiiggeabnkjhpaldj?hl=en-US//). 安装后，在某个搜索引擎中查找图像，然后使用之前下载的扩展程序下载图片。
  ![image](images.png)

#### 第一步

在任意搜索引擎中搜索感兴趣的图片主题。

#### 第二步

打开先前安装的扩展程序。

#### 第三步

选择图像类型为JPG。

#### 第四步

选择所有图片并下载。

+ **选项 3**. 需要安装Python库：[yandex-images-download](https://pypi.org/project/yandex-images-download/?msclkid=b0148afab45011ec8358c9751dabcf63//) ([GitHub](https://github.com/doevent/yandex-images-downloader/?msclkid=b0155486b45011eca4a25458cfa90a0e//)). 安装库后，需要下载适合您版本的Google Chrome的 [ChromeDriver](https://chromedriver.chromium.org/?msclkid=c622b0f0b45011ec8c6768a6d02ae314//) ChromeDriver，并解压文件 chromedriver.exe. 在PyCharm的命令行中输入以下命令： *yandex-images-download Chrome --keywords "Торт, Ласточка, Кошка" --limit 100* (另一种命令示例： *yandex-images-download Chrome --keywords "ласточка, орёл, торты" -o путь_для_скачивания -l количество фотографий -dp путь_к_chromedriver.exe*).

### 图像验证

成功下载图像后，需要手动清理多余的对象，以排除可能的错误（重复、错误<mark>图像</mark>）。

### 将图像导入到Google Colab

#### 第一步

需要点击链接进入预先准备的项目[ipynb](homework1-中文.ipynb) 并将其副本保存到您的Google云端硬盘。

#### 第二步

在Colab文件系统中的“content”文件夹里，根据您图像类别的主题创建三个文件夹（例如：“蛋糕”、“燕子”、“猫”），并将之前下载的图像上传到这些文件夹中。
![image](colab.png)

在Colab中打开文件系统（将自动打开“content”文件夹）。右键单击空白区域，并根据类别主题创建文件夹。

#### 第三步

将类别变量修改为您自己的<mark>类别</mark>。

### 在Colab中优化原始模型

需要进行数据增强、正则化和迁移学习，并将模型保存为ONNX格式。

## 第2部分.图像分类Web应用程序

## 安装IDE VS Code

为完成作业需要安装 [VS Code](https://code.visualstudio.com).

### 创建用于分类获得的数据集图像的Web应用程序

#### 第一步

在VS Code中创建Django项目的示例可以通过此链接[ссылке](https://github.com/iu5git/Web/blob/main/tutorials/lab1-py/lab1_tutorial.md)查看。

#### 第二步

创建项目后，需要在项目根目录中创建一个名为media的文件夹，以便后续保存图像和ONNX格式文件。在media文件夹内创建“images”和“models”文件夹。

![image](folders.png)

#### 第三步

在settings.py文件中添加到先前创建的media文件夹的路径。在文件末尾添加以下代码块：

```python
import os

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'media')
```

为了让Django知道在哪里查找我们的<mark>网页模板</mark>，请检查文件settings.py。在变量TEMPLATES中的DIRS字段中指定您的<mark>网页模板</mark>路径。

```python
TEMPLATES = [
    {
        # ...
        "DIRS": [BASE_DIR / "DZ1/templates"],
        # ...
    },
]
```

#### 第四步

在与settings.py文件相同的文件夹中添加一个名为views.py的Python文件。

![image](views.png)

```python
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import onnxruntime
import numpy as np
from PIL import Image

imageClassList = {'0': 'Торт', '1': 'Ласточка', '2': 'Кошка'}  #此处标明类别

def scoreImagePage(request):
    return render(request, 'scorepage.html')

def predictImage(request):
    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save('images/'+fileObj.name,fileObj)
    filePathName = fs.url(filePathName)
    modelName = request.POST.get('modelName')
    scorePrediction = predictImageData(modelName, '.'+filePathName)
    context = {'scorePrediction': scorePrediction}
    return render(request, 'scorepage.html', context)

def predictImageData(modelName, filePath):
    img = Image.open(filePath).convert("RGB")
    img = np.asarray(img.resize((32, 32), Image.ANTIALIAS))
    sess = onnxruntime.InferenceSession(r'C:\DZ1\media\models\cifar100.onnx') #<-此处必填указать свой путь к модели
    outputOFModel = np.argmax(sess.run(None, {'input': np.asarray([img]).astype(np.float32)}))
    score = imageClassList[str(outputOFModel)]
    return score
```

##### 第4.1步

在views.py文件中，根据您的主题修改类别，并指定ONNX模型的路径。

```python
imageClassList = {'0': 'Торт', '1': 'Ласточка', '2': 'Кошка'}  #在此指定类别
```

```python
sess = onnxruntime.InferenceSession(r'C:\DZ1\media\models\cifar100.onnx') #<-在这里您需要指定模型的路径
```

#### 第五步

在urls.py文件中，将内容替换为以下代码块：

```python
from django.contrib import admin
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from . import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.scoreImagePage, name='scoreImagePage'),
    path('predictImage', views.predictImage, name='predictImage'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

#### 第六步

安装以下库：onnx, onnxruntime, numpy, pillow。
可以使用以下命令在终端中安装每个库：
![image](shell.png)

#### 第七步

在templates文件夹中添加一个名为scorepage.html的文件。

```html
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="windows-1251">
<title>Домашнее задание 1</title>
<style>
    /* 在文档主体上添加一些填充，以防止内容进入页眉和页脚下方 */
    body{        
        padding-top: 60px;
        padding-bottom: 40px;
    }
    .fixed-header, .fixed-footer{
        width: 100%;
        position: fixed;        
        background: #333;
        padding: 10px 0;
        color: #fff;
    }
    .fixed-header{
        top: 0;
    }
    .fixed-footer{
        bottom: 0;
    }
    .container{
        width: 80%;
        margin: 0 auto; /* 将 DIV 水平居中 */
    }
    nav a{
        color: #fff;
        text-decoration: none;
        padding: 7px 25px;
        display: inline-block;
    }
</style>
</head>
<body>
    <div class="fixed-header">
        <div class="container">

        </div>
    </div>
    <div class="container">
        <form action="predictImage" method="post" enctype="multipart/form-data">
            {% csrf_token %}

            <div class="col-md-4 col-sm-4">
                <label for="FilePath">Загрузить изображение:</label>
            </div> <input name="filePath" type="file"><br><br>
            <input type="submit" value="Submit" >
        </form>
    </div>    
    <div>
        <br>
        {% if scorePrediction %}
        <h3>The classification is : {{scorePrediction}}</h3>
        {% endif %}

    </div>
    <div class="fixed-footer">
        <div class="container"></div>        
    </div>
</body>
</html>
```

#### 第八步

在PyCharm的终端中运行以下命令以启动项目："python3 manage.py runserver".

#### 第九步

上传图像并点击“submit”按钮.

![image](page.png)
