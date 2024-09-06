## 步骤1. 构建应用程序。
1) 创建文件夹 `your_github_id.github.io`(your_github_id应改为您自己的github账户名),在其中打开终端并运行`npm init –y`.
2) 将`package.json`的内容替换为以下内容：
```
{
  "name": "yolov8seg",
  "homepage": "https://your_github_id.github.io/",
  "version": "0.1.0",
  "dependencies": {
    "@techstark/opencv-js": "4.5.5-release.2",
    "onnxruntime-web": "^1.14.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1"
  },
  "scripts": {
    "start": "craco start",
    "build": "craco build",
    "predeploy": "yarn build",
    "deploy": "gh-pages -d build"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  },
  "devDependencies": {
    "@craco/craco": "^7.1.0",
    "copy-webpack-plugin": "^11.0.0",
    "gh-pages": "^6.1.1"
  }
}
```
3) 在项目的根目录中，创建一个包含以下内容的文件`craco.config.js`：
```
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
  webpack: {
    plugins: {
      add: [
        new CopyPlugin({
          // Use copy plugin to copy *.wasm to output folder.
          patterns: [
            { from: "node_modules/onnxruntime-web/dist/*.wasm", to: "static/js/[name][ext]" },
            { from: './public/model/model.onnx', to: '[name][ext]'},
            { from: './public/model/nms-yolov8.onnx', to: '[name][ext]'},
            { from: './public/model/mask-yolov8-seg.onnx', to: '[name][ext]'}
          ],
        }),
      ],
    },
    configure: (config) => {
      // set resolve.fallback for opencv.js
      config.resolve.fallback = {
        fs: false,
        path: false,
        crypto: false,
      };
      return config;
    },
  },
};
```
4) 检查应用程序是否已构建。在项目根目录中创建一个`src`文件夹。在其中我们创建文件`App.js`：
```
import React from "react";

const App = () => {
    return (
        <div>
            Пример onnx
        </div>
    );
};
export default App;
```
在`src`文件夹中创建文件`index.js`：
```
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```
5) 在项目根目录中创建一个`public`文件夹。在其中我们创建文件`index.html`：
```
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta
      name="description"
      content="Object Segmentation Application using YOLOv8 and ONNXRUNTIME"
    />
    <!--
      manifest.json provides metadata used when your web app is installed on a
      user's mobile device or desktop. See https://developers.google.com/web/fundamentals/web-app-manifest/
    -->
    <!--
      Notice the use of %PUBLIC_URL% in the tags above.
      It will be replaced with the URL of the `public` folder during the build.
      Only files inside the `public` folder can be referenced from the HTML.

      Unlike "/favicon.ico" or "favicon.ico", "%PUBLIC_URL%/favicon.ico" will
      work correctly both with client-side routing and a non-root public URL.
      Learn how to configure a non-root public URL by running `npm run build`.
    -->
    <title>YOLOv8 Object Segmentation App</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
    <!--
      This HTML file is a template.
      If you open it directly in the browser, you will see an empty page.

      You can add webfonts, meta tags, or analytics to this file.
      The build step will place the bundled scripts into the <body> tag.

      To begin the development, run `npm start` or `yarn start`.
      To create a production bundle, use `npm run build` or `yarn build`.
    -->
  </body>
</html>
```
在`public`文件夹中，创建一个`model`文件夹。下载[nms-yolov8.onnx](https://github.com/Cai-Chuqiao/Cai-Chuqiao.github.io/blob/main/public/model/nms-yolov8.onnx)和[mask-yolov8-seg.onnx](https://github.com/Cai-Chuqiao/Cai-Chuqiao.github.io/blob/main/public/model/mask-yolov8-seg.onnx)并将它们放入其中。
6) 在`src`文件夹中，我们创建`components`文件夹。在其中我们创建文件`loader.js`：
```
import React from "react";
import "../style/loader.css";

const Loader = (props) => {
  return (
    <div className="wrapper" {...props}>
      <div className="spinner"></div>
      <p>{props.children}</p>
    </div>
  );
};

export default Loader;
```
7) 在`src`文件夹中，我们创建`style`文件夹。它用于存储页面样式文件。<br>
在其中我们创建文件`App.css`：
```
.App {
  height: 100vh;
  padding: 0 10px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

.App > * {
  margin: 3px 0;
}

.header {
  text-align: center;
}

.header p {
  margin: 5px 0;
}

.code {
  padding: 5px;
  color: greenyellow;
  background-color: black;
  border-radius: 5px;
}

.content > img {
  width: 100%;
  max-width: 720px;
  max-height: 500px;
  border-radius: 10px;
}

.content {
  position: relative;
}

.content > canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

button {
  text-decoration: none;
  color: white;
  background-color: black;
  border: 2px solid black;
  margin: 0 5px;
  padding: 5px;
  border-radius: 5px;
  cursor: pointer;
}

button:hover {
  color: black;
  background-color: white;
  border: 2px solid black;
}
```
在其中我们创建文件`index.css`：
```
* {
  margin: 0;
  padding: 0;
}

body {
  width: 100%;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Oxygen", "Ubuntu",
    "Cantarell", "Fira Sans", "Droid Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, "Courier New", monospace;
}
```
在其中我们创建文件`loader.css`：
```
.wrapper {
  background-color: rgba(255, 255, 255, 0.5);
  position: absolute;
  width: 100%;
  height: 100%;
  z-index: 1000;
  display: flex;
  justify-content: center;
  align-items: center;
  font-size: medium;
}

.wrapper > .spinner {
  width: 40px;
  height: 40px;
  background-color: #333;

  margin: 10px 10px;
  -webkit-animation: sk-rotateplane 1.2s infinite ease-in-out;
  animation: sk-rotateplane 1.2s infinite ease-in-out;
}

@-webkit-keyframes sk-rotateplane {
  0% {
    -webkit-transform: perspective(120px);
  }
  50% {
    -webkit-transform: perspective(120px) rotateY(180deg);
  }
  100% {
    -webkit-transform: perspective(120px) rotateY(180deg) rotateX(180deg);
  }
}

@keyframes sk-rotateplane {
  0% {
    transform: perspective(120px) rotateX(0deg) rotateY(0deg);
    -webkit-transform: perspective(120px) rotateX(0deg) rotateY(0deg);
  }
  50% {
    transform: perspective(120px) rotateX(-180.1deg) rotateY(0deg);
    -webkit-transform: perspective(120px) rotateX(-180.1deg) rotateY(0deg);
  }
  100% {
    transform: perspective(120px) rotateX(-180deg) rotateY(-179.9deg);
    -webkit-transform: perspective(120px) rotateX(-180deg) rotateY(-179.9deg);
  }
}

.wrapper > p {
  margin: 0;
}
```
8) 在`src`文件夹中，我们创建`utils`文件夹。它用来放置一些存有项目所必需的函数的文件。
在其中我们创建文件`detect.js`，它存储用来对yolov8模型输出张量进行后处理的函数：
```
import cv from "@techstark/opencv-js";
import { Tensor } from "onnxruntime-web";
import { renderBoxes, Colors } from "./renderBox";
import labels from "./labels.json";

const colors = new Colors();
const numClass = labels.length;

/**
 * Detect Image
 * @param {HTMLImageElement} image Image to detect
 * @param {HTMLCanvasElement} canvas canvas to draw boxes
 * @param {ort.InferenceSession} session YOLOv8 onnxruntime session
 * @param {Number} topk Integer representing the maximum number of boxes to be selected per class
 * @param {Number} iouThreshold Float representing the threshold for deciding whether boxes overlap too much with respect to IOU
 * @param {Number} scoreThreshold Float representing the threshold for deciding when to remove boxes based on score
 * @param {Number[]} inputShape model input shape. Normally in YOLO model [batch, channels, width, height]
 */
export const detectImage = async (
  image,
  canvas,
  session,
  topk,
  iouThreshold,
  scoreThreshold,
  inputShape
) => {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas

  const [modelWidth, modelHeight] = inputShape.slice(2);
  const maxSize = Math.max(modelWidth, modelHeight); // max size in input model
  const [input, xRatio, yRatio] = preprocessing(image, modelWidth, modelHeight); // preprocess frame

  const tensor = new Tensor("float32", input.data32F, inputShape); // to ort.Tensor
  const config = new Tensor(
    "float32",
    new Float32Array([
      numClass, // num class
      topk, // topk per class
      iouThreshold, // iou threshold
      scoreThreshold, // score threshold
    ])
  ); // nms config tensor
  const { output0, output1 } = await session.net.run({ images: tensor }); // run session and get output layer. out1: detect layer, out2: seg layer
  const { selected } = await session.nms.run({ detection: output0, config: config }); // perform nms and filter boxes

  const boxes = []; // ready to draw boxes
  let overlay = new Tensor("uint8", new Uint8Array(modelHeight * modelWidth * 4), [
    modelHeight,
    modelWidth,
    4,
  ]); // create overlay to draw segmentation object

  // looping through output
  for (let idx = 0; idx < selected.dims[1]; idx++) {
    const data = selected.data.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]); // get rows
    let box = data.slice(0, 4); // det boxes
    const scores = data.slice(4, 4 + numClass); // det classes probability scores
    const score = Math.max(...scores); // maximum probability scores
    const label = scores.indexOf(score); // class id of maximum probability scores
    const color = colors.get(label); // get color

    box = overflowBoxes(
      [
        box[0] - 0.5 * box[2], // before upscale x
        box[1] - 0.5 * box[3], // before upscale y
        box[2], // before upscale w
        box[3], // before upscale h
      ],
      maxSize
    ); // keep boxes in maxSize range

    const [x, y, w, h] = overflowBoxes(
      [
        Math.floor(box[0] * xRatio), // upscale left
        Math.floor(box[1] * yRatio), // upscale top
        Math.floor(box[2] * xRatio), // upscale width
        Math.floor(box[3] * yRatio), // upscale height
      ],
      maxSize
    ); // upscale boxes

    boxes.push({
      label: labels[label],
      probability: score,
      color: color,
      bounding: [x, y, w, h], // upscale box
    }); // update boxes to draw later

    const mask = new Tensor(
      "float32",
      new Float32Array([
        ...box, // original scale box
        ...data.slice(4 + numClass), // mask data
      ])
    ); // mask input
    const maskConfig = new Tensor(
      "float32",
      new Float32Array([
        maxSize,
        x, // upscale x
        y, // upscale y
        w, // upscale width
        h, // upscale height
        ...Colors.hexToRgba(color, 120), // color in RGBA
      ])
    ); // mask config
    const { mask_filter } = await session.mask.run({
      detection: mask,
      mask: output1,
      config: maskConfig,
      overlay: overlay,
    }); // perform post-process to get mask

    overlay = mask_filter; // update overlay with the new one
  }

  const mask_img = new ImageData(new Uint8ClampedArray(overlay.data), modelHeight, modelWidth); // create image data from mask overlay
  ctx.putImageData(mask_img, 0, 0); // put overlay to canvas

  renderBoxes(ctx, boxes); // draw boxes after overlay added to canvas

  input.delete(); // delete unused Mat
};

/**
 * Preprocessing image
 * @param {HTMLImageElement} source image source
 * @param {Number} modelWidth model input width
 * @param {Number} modelHeight model input height
 * @param {Number} stride model stride
 * @return preprocessed image and configs
 */
const preprocessing = (source, modelWidth, modelHeight, stride = 32) => {
  const mat = cv.imread(source); // read from img tag
  const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3); // new image matrix
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR); // RGBA to BGR

  const [w, h] = divStride(stride, matC3.cols, matC3.rows);
  cv.resize(matC3, matC3, new cv.Size(w, h));

  // padding image to [n x n] dim
  const maxSize = Math.max(matC3.rows, matC3.cols); // get max size from width and height
  const xPad = maxSize - matC3.cols, // set xPadding
    xRatio = maxSize / matC3.cols; // set xRatio
  const yPad = maxSize - matC3.rows, // set yPadding
    yRatio = maxSize / matC3.rows; // set yRatio
  const matPad = new cv.Mat(); // new mat for padded image
  cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT); // padding black

  const input = cv.blobFromImage(
    matPad,
    1 / 255.0, // normalize
    new cv.Size(modelWidth, modelHeight), // resize to model input size
    new cv.Scalar(0, 0, 0),
    true, // swapRB
    false // crop
  ); // preprocessing image matrix

  // release mat opencv
  mat.delete();
  matC3.delete();
  matPad.delete();

  return [input, xRatio, yRatio];
};

/**
 * Get divisible image size by stride
 * @param {Number} stride
 * @param {Number} width
 * @param {Number} height
 * @returns {Number[2]} image size [w, h]
 */
const divStride = (stride, width, height) => {
  if (width % stride !== 0) {
    if (width % stride >= stride / 2) width = (Math.floor(width / stride) + 1) * stride;
    else width = Math.floor(width / stride) * stride;
  }
  if (height % stride !== 0) {
    if (height % stride >= stride / 2) height = (Math.floor(height / stride) + 1) * stride;
    else height = Math.floor(height / stride) * stride;
  }
  return [width, height];
};

/**
 * Handle overflow boxes based on maxSize
 * @param {Number[4]} box box in [x, y, w, h] format
 * @param {Number} maxSize
 * @returns non overflow boxes
 */
const overflowBoxes = (box, maxSize) => {
  box[0] = box[0] >= 0 ? box[0] : 0;
  box[1] = box[1] >= 0 ? box[1] : 0;
  box[2] = box[0] + box[2] <= maxSize ? box[2] : maxSize - box[0];
  box[3] = box[1] + box[3] <= maxSize ? box[3] : maxSize - box[1];
  return box;
};
```
在其中我们创建文件`renderBox.js`，它用来存储bounding box的渲染函数：
```
/**
 * Render prediction boxes
 * @param {HTMLCanvasElement} canvas canvas tag reference
 * @param {Array[Object]} boxes boxes array
 */
export const renderBoxes = (ctx, boxes) => {
  // font configs
  const font = `${Math.max(
    Math.round(Math.max(ctx.canvas.width, ctx.canvas.height) / 40),
    14
  )}px Arial`;
  ctx.font = font;
  ctx.textBaseline = "top";

  boxes.forEach((box) => {
    const klass = box.label;
    const color = box.color;
    const score = (box.probability * 100).toFixed(1);
    const [x1, y1, width, height] = box.bounding;

    // draw border box
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(Math.min(ctx.canvas.width, ctx.canvas.height) / 200, 2.5);
    ctx.strokeRect(x1, y1, width, height);

    // draw the label background.
    ctx.fillStyle = color;
    const textWidth = ctx.measureText(klass + " - " + score + "%").width;
    const textHeight = parseInt(font, 10); // base 10
    const yText = y1 - (textHeight + ctx.lineWidth);
    ctx.fillRect(
      x1 - 1,
      yText < 0 ? 0 : yText,
      textWidth + ctx.lineWidth,
      textHeight + ctx.lineWidth
    );

    // Draw labels
    ctx.fillStyle = "#ffffff";
    ctx.fillText(klass + " - " + score + "%", x1 - 1, yText < 0 ? 1 : yText + 1);
  });
};

export class Colors {
  // ultralytics color palette https://ultralytics.com/
  constructor() {
    this.palette = [
      "#FF3838",
      "#FF9D97",
      "#FF701F",
      "#FFB21D",
      "#CFD231",
      "#48F90A",
      "#92CC17",
      "#3DDB86",
      "#1A9334",
      "#00D4BB",
      "#2C99A8",
      "#00C2FF",
      "#344593",
      "#6473FF",
      "#0018EC",
      "#8438FF",
      "#520085",
      "#CB38FF",
      "#FF95C8",
      "#FF37C7",
    ];
    this.n = this.palette.length;
  }

  get = (i) => this.palette[Math.floor(i) % this.n];

  static hexToRgba = (hex, alpha) => {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16), alpha]
      : null;
  };
}
```
在其中我们创建一个文件`labels.json`，它存储需要识别的类：
```
[
  "airplane",
  "weather_balloon",
  "UFO"
]
```
9) 为了对图片进行分析，有必要准备输入数据，即将图像转换为张量。为此，您需要对加载的图像运行预处理函数并将其转换为张量。输入张量经过模型计算获得输出后，需要对输出张量进行后处理，以便在图像上绘制识别结果。因此，您需要将以下代码添加到`App.js`中：
```
import React, { useState, useRef } from "react";
import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/loader";
import { detectImage } from "./utils/detect";
import "./style/App.css";

const App = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState({ text: "Loading OpenCV.js", progress: null });
  const [image, setImage] = useState(null);
  const inputImage = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);

  // configs
  const modelName = "model.onnx";
  const modelInputShape = [1, 3, 640, 640];
  const topk = 100;
  const iouThreshold = 0.45;
  const scoreThreshold = 0.25;

  // wait until opencv.js initialized
  cv["onRuntimeInitialized"] = async () => {
    // create session
    setLoading({ text: "Loading model...", progress: null });
    const yolov8 = await InferenceSession.create('./model.onnx');

    setLoading({ text: "Warming up nms...", progress: null });
    const nms = await InferenceSession.create('./nms-yolov8.onnx');

    setLoading({ text: "Warming up mask...", progress: null });
    const mask = await InferenceSession.create('./mask-yolov8-seg.onnx');

    // warmup main model
    setLoading({ text: "Warming up model...", progress: null });
    const tensor = new Tensor(
      "float32",
      new Float32Array(modelInputShape.reduce((a, b) => a * b)),
      modelInputShape
    );
    await yolov8.run({ images: tensor });

    setSession({ net: yolov8, nms: nms, mask: mask });
    setLoading(null);
  };

  return (
    <div className="App">
      {loading && (
        <Loader>
          {loading.progress ? `${loading.text} - ${loading.progress}%` : loading.text}
        </Loader>
      )}
      <div className="header">
        <h1>YOLOv8 Object Segmentation App</h1>
        <p>
          YOLOv8 object detection application live on browser powered by{" "}
          <code>onnxruntime-web</code>
        </p>
        <p>
          Serving : <code className="code">{modelName}</code>
        </p>
      </div>

      <div className="content">
        <img
          ref={imageRef}
          src="#"
          alt=""
          style={{ display: image ? "block" : "none" }}
          onLoad={() => {
            detectImage(
              imageRef.current,
              canvasRef.current,
              session,
              topk,
              iouThreshold,
              scoreThreshold,
              modelInputShape
            );
          }}
        />
        <canvas
          id="canvas"
          width={modelInputShape[2]}
          height={modelInputShape[3]}
          ref={canvasRef}
        />
      </div>

      <input
        type="file"
        ref={inputImage}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          // handle next image to detect
          if (image) {
            URL.revokeObjectURL(image);
            setImage(null);
          }

          const url = URL.createObjectURL(e.target.files[0]); // create image url
          imageRef.current.src = url; // set image source
          setImage(url);
        }}
      />
      <div className="btn-container">
        <button
          onClick={() => {
            inputImage.current.click();
          }}
        >
          Open local image
        </button>
        {image && (
          /* show close btn when there is image */
          <button
            onClick={() => {
              inputImage.current.value = "";
              imageRef.current.src = "#";
              URL.revokeObjectURL(image);
              setImage(null);
            }}
          >
            Close image
          </button>
        )}
      </div>
    </div>
  );
};
export default App;
```
10) 下载[yarn.lock](https://github.com/Cai-Chuqiao/Cai-Chuqiao.github.io/blob/main/yarn.lock)文件并将其放在根文件夹中。 yarn.lock文件可以保证不同环境下安装的依赖的版本一致。这有助于避免由不兼容的依赖项版本引起的潜在问题和错误。<br>

React项目的程序基于[Hyuto/yolov8-seg-onnxruntime-web](https://github.com/Hyuto/yolov8-seg-onnxruntime-web/tree/master)<br>

## 步骤2. 本地部署 React 项目
### 步骤2.1
将您在[yolov8+onnx.ipynb](https://github.com/Cai-Chuqiao/Cai-Chuqiao.github.io/blob/main/colab/yolov8%2Bonnx.ipynb)中导出的ONNX模型文件放置到`./public/model/`里.
### 步骤2.2
在模板文件`craco.config.js`中将`model.onnx`更改为您自己的 ONNX 文件名。<br>
```
          patterns: [
            { from: "node_modules/onnxruntime-web/dist/*.wasm", to: "static/js/[name][ext]" },
            { from: './public/model/model.onnx', to: '[name][ext]'},
            { from: './public/model/nms-yolov8.onnx', to: '[name][ext]'},
            { from: './public/model/mask-yolov8-seg.onnx', to: '[name][ext]'}
          ],
```
将`./src/utils/label.json`中的标签更改为您自己的模型识别的类。<br>
```json
[
  "airplane",
  "weather_balloon",
  "UFO"
]
```
### 步骤2.3
如果您只想在本地部署React项目，可以按照[Hyuto/yolov8-seg-onnxruntime-web](https://github.com/Hyuto/yolov8-seg-onnxruntime-web/tree/master)中的步骤进行操作，
但如果你想将 React 项目部署到 GitHub 页面，则暂时不需要执行 `yarn install`、`yarn start` 和 `yarn build`。
否则，在将项目上传到自己的github仓库时，会因为node_modules中的文件太大而导致上传失败。<br>
## 步骤3. 将 React 项目部署到 github pages
### 步骤3.1
在 github 上创建您的存储库。存储库名称为`your_github_id.github.io`(your_github_id应改为您的GitHub用户名)，并且存储库必须是公共的。
![image](https://github.com/Cai-Chuqiao/Cai-Chuqiao.github.io/assets/150286732/7ddf6e33-b88a-45e6-87c5-ee425e079a72)
### 步骤3.2
将本地存储React项目根目录下 `package.json` 中的 `"homepage": "https://your_github_id.github.io/"` 更改为您自己的github pages的URL。
![image](https://github.com/Cai-Chuqiao/Cai-Chuqiao.github.io/assets/150286732/b4ef31b9-d7d8-4d11-b93e-f1efa07e1305)
### 步骤3.3
使用`git pull`将本地项目拉入您的存储库。<br>
### 步骤3.4
在本地 React 项目的根目录中打开终端并依次运行：<br>
```
yarn add gh-pages --save-dev
git add .
git commit -m "deploy commit"
git push
yarn deploy
```
### 步骤3.5
在存储库的`Settings`中选择`pages`，将`Branch`更改为`gh-pages`，然后点击`Save`，
![image](https://github.com/Cai-Chuqiao/Cai-Chuqiao.github.io/assets/150286732/d1456289-e90f-4af4-b16b-38bbdae2c04c)
然后，您就可以通过打开GitHub pages的URL来在线运行React项目了。
![image](https://github.com/Cai-Chuqiao/Cai-Chuqiao.github.io/assets/150286732/3bf1d3f5-3499-4796-b0c9-4b95df9f6d9b)

