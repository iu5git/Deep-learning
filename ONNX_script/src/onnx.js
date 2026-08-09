class ModelManager {

    constructor() {
        this.session = null;

        this.inputName = null;
        this.outputName = null;

        this.inputMetadata = null;
        this.outputMetadata = null;

        this.loaded = false;
    }

    async loadModel(file) {

        if (!file) {
            throw new Error("Файл модели не выбран");
        }

        const buffer = await file.arrayBuffer();

        try {
            this.session = await ort.InferenceSession.create(
                buffer,
                {
                    executionProviders: ["cpu"]  // ["wasm"] // виртуальная
                }
            );
        } catch (err) {
            throw new Error(
                `Не удалось загрузить ONNX-модель: ${err.message}`
            );
        }

        if (!this.session.inputNames?.length) {
            throw new Error("У модели отсутствуют входы");
        }

        if (!this.session.outputNames?.length) {
            throw new Error("У модели отсутствуют выходы");
        }

        this.inputName = this.session.inputNames[0];
        this.outputName = this.session.outputNames[0];

        console.log("== MODEL LOAD ==")
        console.log(this.session.inputNames);
        console.log(this.session.inputMetadata);

        // =========================
        // SAFE METADATA HANDLING
        // =========================

        this.inputMetadata =
            this.session.inputMetadata?.[this.inputName] ||
            Object.values(this.session.inputMetadata || {})[0] ||
            null;

        this.outputMetadata =
            this.session.outputMetadata?.[this.outputName] ||
            Object.values(this.session.outputMetadata || {})[0] ||
            null;

        this.loaded = true;

        return this.getModelInfo();
    }

    getModelInfo() {

        if (!this.loaded) {
            return null;
        }

        console.log("ONNX input metadata:", this.session.inputMetadata);
        console.log("ONNX output metadata:", this.session.outputMetadata);

        return {
            inputName: this.inputName,
            outputName: this.outputName,

            inputType: this.inputMetadata?.type || "float32",
            outputType: this.outputMetadata?.type || "float32",

            inputDimensions: this.inputMetadata?.dimensions ?? [],
            outputDimensions: this.outputMetadata?.dimensions ?? []
        };
    }

    // =========================
    // OUTPUT SIZE DETECTION
    // =========================

    getOutputCount() {

        if (!this.loaded) return 0;

        const dims = this.outputMetadata?.dimensions || [];

        // ищем последний валидный размер
        for (let i = dims.length - 1; i >= 0; i--) {
            const d = dims[i];

            if (typeof d === "number" && d > 1) {
                return d;
            }
        }

        return 0;
    }

    // =========================
    // INPUT VALIDATION
    // =========================

    validateInputShape(expectedShape) {

        if (!this.loaded) {
            throw new Error("Модель не загружена");
        }

        const actual = this.inputMetadata?.dimensions ?? [];

        if (actual.length !== expectedShape.length) {
            return {
                ok: false,
                message:
                    `Модель ожидает ${actual.length} измерений, ` +
                    `а пользователь указал ${expectedShape.length}`
            };
        }

        for (let i = 0; i < actual.length; i++) {

            const modelDim = actual[i];
            const userDim = expectedShape[i];

            // dynamic axis (-1) игнорируем
            if (typeof modelDim !== "number" || modelDim === -1) {
                continue;
            }

            if (modelDim !== userDim) {
                return {
                    ok: false,
                    message:
                        `Несовпадение размерности по оси ${i}: ` +
                        `${modelDim} ≠ ${userDim}`
                };
            }
        }

        return {
            ok: true,
            message: "OK"
        };
    }

    // =========================
    // INFERENCE
    // =========================

    async predict(inputTensor) {

        if (!this.loaded) {
            throw new Error("Модель не загружена");
        }

        if (!(inputTensor instanceof ort.Tensor)) {
            throw new Error("В predict() должен передаваться ort.Tensor");
        }

        const feeds = {
            [this.inputName]: inputTensor
        };

        console.log("tensor shape:", inputTensor.dims);
        console.log("tensor sample:", inputTensor.data.slice(0, 10));

        let results;

        try {
            results = await this.session.run(feeds);
        } catch (err) {
            throw new Error(
                `Ошибка инференса: ${err.message}`
            );
        }

        const outputTensor =
            results[this.outputName] ||
            Object.values(results)[0];

        if (!outputTensor) {
            throw new Error("Выходной тензор отсутствует");
        }

        return outputTensor;
    }

    async predictArray(inputTensor) {

        const output = await this.predict(inputTensor);
        return Array.from(output.data);
    }

    // =========================
    // POSTPROCESSING
    // =========================

    softmax(logits) {

        if (!logits?.length) {
            return [];
        }

        const maxValue = Math.max(...logits);

        const exps = logits.map(
            x => Math.exp(x - maxValue)
        );

        const sum = exps.reduce(
            (a, b) => a + b,
            0
        );

        return exps.map(
            x => x / sum
        );
    }

    getTopClass(scores) {

        if (!scores?.length) {
            return {
                index: -1,
                value: 0
            };
        }

        let bestIndex = 0;
        let bestValue = scores[0];

        for (let i = 1; i < scores.length; i++) {
            if (scores[i] > bestValue) {
                bestValue = scores[i];
                bestIndex = i;
            }
        }

        return {
            index: bestIndex,
            value: bestValue
        };
    }

    // =========================
    // CLEANUP
    // =========================

    dispose() {

        try {
            this.session?.release?.();
        } catch (_) {
            // ignore
        }
        
        this.session?.endProfiling?.();
        this.session = null;

        this.inputName = null;
        this.outputName = null;

        this.inputMetadata = null;
        this.outputMetadata = null;

        this.loaded = false;
    }
}

window.modelManager = new ModelManager();