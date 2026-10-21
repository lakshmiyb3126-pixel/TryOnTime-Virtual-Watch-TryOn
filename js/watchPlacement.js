
/*
 * Watch Placement Module
 * Kavana S - Virtual Watch Try-On
 * Uses JavaScript Canvas API
 */

function loadImage(source) {
    return new Promise((resolve, reject) => {
        const image = new Image();

        image.onload = () => resolve(image);
        image.onerror = () => reject(
            new Error("Unable to load image: " + source)
        );

        image.src = source;
    });
}

async function placeWatch(canvas, wristSource, watchSource, wristData) {
    const wristImage = await loadImage(wristSource);
    const watchImage = await loadImage(watchSource);

    const ctx = canvas.getContext("2d");

    if (!ctx) {
        throw new Error("Canvas is not supported.");
    }

    if (!wristData ||
        !Number.isFinite(wristData.x) ||
        !Number.isFinite(wristData.y) ||
        !Number.isFinite(wristData.width) ||
        !Number.isFinite(wristData.angle) ||
        wristData.width <= 0) {
        throw new Error("Invalid wrist coordinates.");
    }

    // Match canvas size to original wrist photograph.
    canvas.width = wristImage.naturalWidth;
    canvas.height = wristImage.naturalHeight;

    // Draw original wrist photograph.
    ctx.drawImage(wristImage, 0, 0);

    // Size the watch according to wrist width.
    const watchWidth = wristData.width * 1.05;
    const watchHeight =
        watchWidth * (watchImage.naturalHeight / watchImage.naturalWidth);

    // Position the watch at the wrist center.
    ctx.save();
    ctx.translate(wristData.x, wristData.y);

    // Angle is measured in radians.
    ctx.rotate(wristData.angle);

    // Draw watch centered at wrist position.
    ctx.drawImage(
        watchImage,
        -watchWidth / 2,
        -watchHeight / 2,
        watchWidth,
        watchHeight
    );

    ctx.restore();

    return canvas;
}
