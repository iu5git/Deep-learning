// Настройка ONNX для стабильности
if (typeof ort !== 'undefined') {
    ort.env.wasm.numThreads = 1;
}

let modelReady = false;
let imageReady = false;
let lastTensor = null;
let running = false;

const $ = (id) => document.getElementById(id);

/* =========================
   UI
========================= */

const modelFile = $("modelFile");
const buildBtn = $("buildBtn");

const layoutSelect = $("layoutSelect");

const modelStatus = $("modelStatus");

const inputNameEl = $("inputName");
const outputNameEl = $("outputName");
const tensorTypeEl = $("tensorType");
const tensorDimsEl = $("tensorDims");

const outputCountInfo = $("outputCountInfo");

const labelsSection = $("labelsSection");
const imageSection = $("imageSection");

const labelId = $("labelId");
const addBtn = $("addBtn");
const undoBtn = $("undoBtn");
const resetBtn = $("resetBtn");

const assignedLabels = $("assignedLabels");

const imageFile = $("imageFile");

const imageWidth = $("imageWidth");
const imageHeight = $("imageHeight");
const shiftStep = $("shiftStep");

const workingSize = $("workingSize");
const imageWarning = $("imageWarning");

const offsetXValue = $("offsetXValue");
const offsetYValue = $("offsetYValue");

const prediction = $("prediction");
const topPrediction = $("topPrediction");

const histogram = $("histogram");
const hctx = histogram.getContext("2d");

/* =========================
   STATE
========================= */

const labels = [];

/* =========================
   MODEL LOAD
========================= */

buildBtn.onclick = async () => {

    const file = modelFile.files[0];

    if (!file) {
        modelStatus.textContent = "❌ Файл модели не выбран";
        return;
    }

    modelStatus.textContent = "Загрузка модели...";

    try {
        await ort.env.wasm?.reset?.();

        modelManager.dispose();
        modelReady = false;
 
        const info = await modelManager.loadModel(file);

        modelReady = true;
        lastTensor = null;
        running = false;

        inputNameEl.textContent = info.inputName;
        outputNameEl.textContent = info.outputName;
        tensorTypeEl.textContent = info.inputType;

        tensorDimsEl.textContent =
            JSON.stringify(info.inputDimensions);

        const outCount = modelManager.getOutputCount();

        outputCountInfo.textContent =
            `Выходов: ${outCount || "?"}`;

        modelStatus.textContent = "✓ модель загружена";

        labelsSection.classList.remove("hidden");
        imageSection.classList.remove("hidden");

        // reset image state
        imageReady = false;
        lastTensor = null;

        imageFile.value = "";

        prediction.textContent = "—";
        topPrediction.textContent = "";

        offsetXValue.textContent = "0";
        offsetYValue.textContent = "0";

    } catch (e) {
        modelStatus.textContent = "❌ " + e.message;
    }
};

/* =========================
   LABELS
========================= */

function renderLabels() {

    console.log("LABEL MAP =", labels);

    assignedLabels.innerHTML = "";

    labels.forEach((l, i) => {

        const li = document.createElement("li");

        const name =
            CIFAR100_NAMES[l] ?? `class_${l}`;

        li.textContent = `${i}: ${name} (${l})`;

        assignedLabels.appendChild(li);
    });
}

addBtn.onclick = () => {
    const v = parseInt(labelId.value);
    if (isNaN(v)) return;

    labels.push(v);
    renderLabels();
    runPipeline();
};

undoBtn.onclick = () => {
    labels.pop();
    renderLabels();
    runPipeline();
};

resetBtn.onclick = () => {
    labels.length = 0;
    renderLabels();
    runPipeline();
};

document.getElementById("generateRandomBtn").onclick = () => {

    labels.length = 0;

    const max = 100;

    for (let i = 0; i < 10; i++) {
        labels.push(Math.floor(Math.random() * max));
    }

    renderLabels();
    runPipeline();
};

/* =========================
   IMAGE LOAD
========================= */

imageFile.onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    try {
        syncImageConfig();        // 1. сначала конфиг

        await imageManager.load(file);  // 2. загрузка + render внутри

        imageReady = true;
        runPipeline();

    } catch (err) {
        imageWarning.textContent =
            "Ошибка загрузки изображения: " + err.message;
    }
};

function syncImageConfig() {

    imageManager.setConfig({
        width: parseInt(imageWidth.value),
        height: parseInt(imageHeight.value),
        shiftStep: Number(shiftStep.value) || 1,
        layout: layoutSelect.value
    });

    workingSize.textContent =
        `${imageWidth.value} × ${imageHeight.value}`;
}

/* =========================
   SHIFT
========================= */

function shift(dx, dy) {

     if (!imageManager.image) {
        return;
    }

    imageManager.shift(dx, dy);

    offsetXValue.textContent = imageManager.offsetX;
    offsetYValue.textContent = imageManager.offsetY;

    runPipeline();
}

window.shift = shift;

document.getElementById("shiftLeftBtn").onclick = () => shift(-1, 0);
document.getElementById("shiftRightBtn").onclick = () => shift(1, 0);
document.getElementById("shiftUpBtn").onclick = () => shift(0, -1);
document.getElementById("shiftDownBtn").onclick = () => shift(0, 1);

document.getElementById("resetShiftBtn").onclick = () => {

    imageManager.resetShift();

    offsetXValue.textContent = 0;
    offsetYValue.textContent = 0;

    runPipeline();
};

document.addEventListener("keydown", (e) => {

    switch (e.key) {
        case "ArrowLeft":
            shift(-1, 0);
            break;

        case "ArrowRight":
            shift(1, 0);
            break;

        case "ArrowUp":
            shift(0, -1);
            break;

        case "ArrowDown":
            shift(0, 1);
            break;

        case " ":
        case "Spacebar": // для старых браузеров

            e.preventDefault();

            if (e.shiftKey) {
                imageManager.shiftStep = Math.max(1, imageManager.shiftStep - 1);
            } else {
                imageManager.shiftStep += 1;
            }
            const shiftStepInput = document.getElementById("shiftStep");
            shiftStepInput.value = imageManager.shiftStep;
            return;

        default:
            return;
    }

    e.preventDefault();
});

/* =========================
   PIPELINE
========================= */

async function runPipeline() {

    if (!modelReady || !imageReady) return;
    if (running) return;

    running = true;

    try {

        const tensor = imageManager.toTensor();
        lastTensor = null;

        console.log("=== INPUT TENSOR DEBUG ===");
        console.log("shape:", tensor.dims);
        console.log("first values:", tensor.data.slice(0, 20));
        console.log("min/max:", Math.min(...tensor.data), Math.max(...tensor.data));

        const logits =
            await modelManager.predictArray(tensor);

        const probs =
            modelManager.softmax(logits);

        const top =
            modelManager.getTopClass(probs);

        console.log("=== MODEL DEBUG ===");
        console.log("logits:", logits);
        console.log("probs:", probs);
        console.log("top:", top);
        console.log("sum probs:", probs.reduce((a,b)=>a+b,0));
        console.log("max prob:", Math.max(...probs));

        renderPrediction(top);
        drawHistogram(probs);

    } catch (e) {

        prediction.textContent =
            "Ошибка: " + e.message;

    } finally {
        running = false;
        if (window.gc) window.gc();
    }
}

/* =========================
   OUTPUT
========================= */

function renderPrediction(top) {

    const modelClass = top.index;

    const label =
        labels.length > 0
            ? (CIFAR100_NAMES[labels[modelClass]] ?? `class_${modelClass}`)
            : `class ${modelClass}`;

    prediction.textContent =
        `${label} — ${(top.value * 100).toFixed(2)}%`;

    topPrediction.textContent =
        `Top-1: ${label}`;
}

/* =========================
   HISTOGRAM (FIXED)
========================= */

function drawHistogram(preds) {

    if (!preds?.length) return;

    const ctx = hctx;
    const w = histogram.width;
    const h = histogram.height;

    ctx.clearRect(0, 0, w, h);

    // =========================
    // LAYOUT CONFIG
    // =========================
    const bottomPad = 60;   // зона под подписи
    const topPad = 20;      // зона под grid labels
    const chartH = h - bottomPad - topPad;

    const step = w / preds.length;

    // =========================
    // GRID (0..100%)
    // =========================
    ctx.strokeStyle = "#eee";
    ctx.fillStyle = "#666";
    ctx.font = "14px sans-serif";
    ctx.textAlign = "right";

    for (let p = 0; p <= 100; p += 10) {

        const y = topPad + chartH - (p / 100) * chartH;

        // line
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();

        // label right axis
        ctx.fillText(p + "%", w - 5, y - 2);
    }

    // =========================
    // BARS
    // =========================
    for (let i = 0; i < preds.length; i++) {

        const x = i * step;

        const value = preds[i];
        const barH = value * chartH;

        const y = topPad + (chartH - barH);

        ctx.fillStyle = "#4a90e2";
        ctx.fillRect(x + 2, y, step - 4, barH);

        // =========================
        // LABEL (CENTERED, BELOW)
        // =========================
        const classId = labels[i];

        const label =
            classId !== undefined
                ? (CIFAR100_NAMES[classId] ?? `class_${classId}`)
                : `out_${i}`;

        ctx.fillStyle = "#000";
        ctx.font = "11px sans-serif";
        ctx.textAlign = "center";

        ctx.fillText(
            label,
            x + step / 2,
            h - 20
        );
    }
}

/* =========================
   LIVE UPDATE
========================= */

[imageWidth, imageHeight, shiftStep].forEach(el => {

    el.addEventListener("change", () => {

        if (!imageReady) return;

        syncImageConfig();
        imageManager.render();
        runPipeline();
    });
});

/* =========================
   CIFAR
========================= */

const CIFAR100_NAMES = [
"apple","aquarium_fish","baby","bear","beaver",
"bed","bee","beetle","bicycle","bottle",
"bowl","boy","bridge","bus","butterfly",
"camel","can","castle","caterpillar","cattle",
"chair","chimpanzee","clock","cloud","cockroach",
"couch","crab","crocodile","cup","dinosaur",
"dolphin","elephant","flatfish","forest","fox",
"girl","hamster","house","kangaroo","keyboard",
"lamp","lawn_mower","leopard","lion","lizard",
"lobster","man","maple_tree","motorcycle","mountain",
"mouse","mushroom","oak_tree","orange","orchid",
"otter","palm_tree","pear","pickup_truck","pine_tree",
"plain","plate","poppy","porcupine","possum",
"rabbit","raccoon","ray","road","rocket",
"rose","sea","seal","shark","shrew",
"skunk","skyscraper","snail","snake","spider",
"squirrel","streetcar","sunflower","sweet_pepper","table",
"tank","telephone","television","tiger","tractor",
"train","trout","tulip","turtle","wardrobe",
"whale","willow_tree","wolf","woman","worm"
];

/* =========================
   POPUP СПИСОК КЛАССОВ
========================= */

// Открыть модальное окно
document.getElementById("showClassesBtn").addEventListener('click', function() {
    const modal = document.getElementById("classesModal");
    modal.style.display = "flex";
    renderClassTable();
    document.getElementById("classSearchInput").value = ""; // Очищаем поиск
});

// Закрыть модальное окно - ИСПРАВЛЕНО
document.getElementById("closeModalBtn").addEventListener('click', function() {
    document.getElementById("classesModal").style.display = "none";
});

// Закрыть по клику вне модального окна
window.addEventListener('click', function(event) {
    const modal = document.getElementById("classesModal");
    if (event.target === modal) {
        modal.style.display = "none";
    }
});

// Поиск по классам - ИСПРАВЛЕНО
document.getElementById("classSearchInput").addEventListener('input', function() {
    renderClassTable(this.value.toLowerCase());
});

// Рендер таблицы классов
function renderClassTable(searchTerm = "") {
    const tbody = document.getElementById("classTableBody");
    tbody.innerHTML = "";
    
    CIFAR100_NAMES.forEach((name, index) => {
        // ИСПРАВЛЕНО: поиск по индексу и названию
        const match = !searchTerm || 
                      name.toLowerCase().includes(searchTerm) || 
                      index.toString().includes(searchTerm);
        if (!match) return;
        
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td style="font-weight: bold; color: #4a90e2; padding: 6px 8px; border-bottom: 1px solid #eee;">${index}</td>
            <td style="padding: 6px 8px; border-bottom: 1px solid #eee;">${name}</td>
        `;
        tr.style.cursor = "pointer";
        tr.addEventListener('click', function() {
            document.getElementById("labelId").value = index;
            document.getElementById("classesModal").style.display = "none";
        });
        tbody.appendChild(tr);
    });
}