class ImageManager {

    constructor() {

        this.image = null;

        // sprite 32x32 (сжатый оригинал)
        this.spriteCanvas = document.createElement("canvas");
        this.spriteCtx = this.spriteCanvas.getContext("2d");

        // working canvas (model input)
        this.workingCanvas = document.getElementById("workingCanvas");
        this.workingCtx = this.workingCanvas.getContext("2d");

        // preview
        this.previewCanvas = document.getElementById("previewCanvas");
        this.previewCtx = this.previewCanvas.getContext("2d");

        this.width = 32;
        this.height = 32;

        this.offsetX = 0;
        this.offsetY = 0;

        this.shiftStep = 1;
        this.layout = "NCHW";
    }

    setConfig({ width, height, shiftStep, layout }) {

        this.width = width;
        this.height = height;

        this.shiftStep = Number(shiftStep) || 1;
        this.layout = layout;

        this.workingCanvas.width = width;
        this.workingCanvas.height = height;

        this.spriteCanvas.width = width;
        this.spriteCanvas.height = height;

        this.previewCtx.imageSmoothingEnabled = false;

        if (this.image) {
            this._buildSprite();
            this.render();
        }
    }

    // =========================
    // LOAD + RESIZE TO SPRITE
    // =========================

    async load(file) {

        if (!file) throw new Error("Изображение не выбрано");

        const img = new Image();
        img.src = URL.createObjectURL(file);

        await new Promise((res, rej) => {
            img.onload = res;
            img.onerror = rej;
        });

        this.image = img;

        this._buildSprite();

        this.offsetX = 0;
        this.offsetY = 0;

        this.render();
    }

    _buildSprite() {

        this.spriteCtx.clearRect(0, 0, this.width, this.height);

        this.spriteCtx.imageSmoothingEnabled = true;

        this.spriteCtx.drawImage(
            this.image,
            0, 0,
            this.width,
            this.height
        );
    }
    // =========================
    // SHIFT (sprite movement)
    // =========================

    shift(dx, dy) {

        const step = Number(this.shiftStep) || 1;

        this.offsetX += dx * step;
        this.offsetY += dy * step;

        // clamp inside canvas bounds
        const maxX = this.width;
        const maxY = this.height;

        this.offsetX = Math.max(-maxX, Math.min(maxX, this.offsetX));
        this.offsetY = Math.max(-maxY, Math.min(maxY, this.offsetY));

        this.render();
    }

    resetShift() {
        this.offsetX = 0;
        this.offsetY = 0;
        this.render();
    }

    // =========================
    // RENDER PIPELINE
    // =========================

    render() {

        if (!this.spriteCanvas) return;

        const ctx = this.workingCtx;

        ctx.clearRect(0, 0, this.width, this.height);

        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, this.width, this.height);

        const cx = Math.floor(this.width / 2 - this.width / 2 + this.offsetX);
        const cy = Math.floor(this.height / 2 - this.height / 2 + this.offsetY);

        ctx.drawImage(
            this.spriteCanvas,
            cx,
            cy,
            this.width,
            this.height
        );
        
        console.log("sprite render offset:", this.offsetX, this.offsetY);
        
        this._drawPreview();
    }

    // =========================
    // PREVIEW
    // =========================

    _drawPreview() {

        const ctx = this.previewCtx;

        ctx.clearRect(
            0, 0,
            this.previewCanvas.width,
            this.previewCanvas.height
        );

        ctx.imageSmoothingEnabled = false;

        ctx.drawImage(
            this.workingCanvas,
            0, 0,
            this.width,
            this.height,
            0, 0,
            this.previewCanvas.width,
            this.previewCanvas.height
        );
    }

    // =========================
    // TENSOR
    // =========================

    toTensor() {

        const ctx = this.workingCtx;

        const imgData = ctx.getImageData(
            0, 0,
            this.width,
            this.height
        );

        const data = imgData.data;

        const pixelCount = this.width * this.height;
        const out = new Float32Array(pixelCount * 3);

        for (let p = 0, i = 0; p < data.length; p += 4, i++) {

            let r = data[p];
            let g = data[p + 1];
            let b = data[p + 2];

            if (this.layout === "NCHW") {
                out[i] = r;
                out[i + pixelCount] = g;
                out[i + 2 * pixelCount] = b;
            } else {
                out[i * 3] = r;
                out[i * 3 + 1] = g;
                out[i * 3 + 2] = b;
            }
        }

        console.log("=== IMAGE DEBUG ===");
        console.log("offset:", this.offsetX, this.offsetY);
        console.log("sample pixel:", out.slice(0, 10));

        console.log("LAYOUT IS:", this.layout)
        return new ort.Tensor(
            "float32",
            out,
            this.layout === "NCHW"
                ? [1, 3, this.height, this.width]
                : [1, this.height, this.width, 3]
        );
    }
}

window.imageManager = new ImageManager();