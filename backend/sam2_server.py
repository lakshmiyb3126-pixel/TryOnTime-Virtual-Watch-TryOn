
"""SAM 2 wrist detection and watch background removal."""

import io
import json
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

app = FastAPI(
    title="Try On Time - AI Watch Fitting"
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# =====================================
# LOAD SAM 2 MODEL
# =====================================

@lru_cache(maxsize=1)
def get_model():
    import torch
    from transformers import Sam2Model, Sam2Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    name = "facebook/sam2.1-hiera-tiny"

    processor = Sam2Processor.from_pretrained(name)
    model = Sam2Model.from_pretrained(name).to(device).eval()

    return torch, processor, model, device


# =====================================
# FIND WRIST BOUNDARIES
# =====================================

def cross_section(mask, origin, direction, palm_width, fraction):
    height, width = mask.shape

    d = np.asarray(direction, dtype=float)
    p = np.array([-d[1], d[0]])

    center = (
        np.asarray(origin, dtype=float)
        + d * palm_width * fraction
    )

    def covered(x, y):
        col = int(round(x))
        row = int(round(y))

        return (
            0 <= col < width
            and 0 <= row < height
            and bool(mask[row, col])
        )

    if not covered(*center):
        return None

    distances = []

    for sign in (-1.0, 1.0):
        last_inside = 0.0
        missing = 0

        for radius in np.arange(
            1.5,
            palm_width * 1.3,
            1.5
        ):
            pos = center + p * sign * radius

            if covered(*pos):
                missing = 0
                last_inside = float(radius)
            else:
                missing += 1

                if missing >= 3:
                    break

        if missing < 3 or last_inside < palm_width * 0.10:
            return None

        distances.append(last_inside)

    left, right = distances

    return {
        "width": left + right,
        "center": center + p * (right - left) * 0.5,
        "t": fraction,
    }


# =====================================
# WRIST GEOMETRY FROM AI MASK
# =====================================

def geometry_from_mask(mask, wrist, axis, palm_width):
    d = np.asarray(axis, dtype=float)
    mag = float(np.linalg.norm(d))

    if mag < 0.1:
        return None

    d /= mag

    sections = [
        cross_section(mask, wrist, d, palm_width, t)
        for t in (0.12, 0.24, 0.40, 0.60, 0.78)
    ]

    sections = [
        s for s in sections if s is not None
    ]

    if len(sections) < 3:
        return None

    widths = np.asarray([
        s["width"] for s in sections
    ])

    typical = float(np.median(widths))

    if not 0.48 * palm_width <= typical <= 1.55 * palm_width:
        return None

    if (
        float(np.max(np.abs(widths - typical)))
        > 0.38 * palm_width
    ):
        return None

    centers = np.stack([
        s["center"] for s in sections
    ])

    fractions = np.asarray([
        s["t"] for s in sections
    ])

    t = fractions - fractions.mean()

    vector = (t[:, None] * centers).sum(axis=0)

    mag = float(np.linalg.norm(vector))

    refined_axis = (
        vector / mag
        if mag > 2
        else d
    )

    if float(np.dot(refined_axis, d)) < 0.85:
        refined_axis = d

    nearby = min(
        sections,
        key=lambda s: abs(s["t"] - 0.24)
    )

    return {
        "width": round(typical, 2),
        "center": {
            "x": round(
                float(nearby["center"][0]), 2
            ),
            "y": round(
                float(nearby["center"][1]), 2
            ),
        },
        "dx": round(
            float(refined_axis[0]), 6
        ),
        "dy": round(
            float(refined_axis[1]), 6
        ),
        "sections": len(sections),
    }


def parse_geometry(value, width, height):
    try:
        data = json.loads(value)

        wrist = np.array([
            float(data["wrist"]["x"]),
            float(data["wrist"]["y"])
        ])

        axis = np.array([
            float(data["dx"]),
            float(data["dy"])
        ])

        palm_width = float(data["palmWidth"])

    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid MediaPipe geometry"
        ) from exc

    if not all(
        np.isfinite(v)
        for v in [*wrist, *axis, palm_width]
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid geometry numbers"
        )

    if (
        palm_width < 8 or
        not (
            0 <= wrist[0] < width
            and 0 <= wrist[1] < height
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="Wrist landmark outside image"
        )

    return wrist, axis, palm_width


# =====================================
# API HEALTH CHECK
# =====================================

@app.get("/")
def health():
    return {
        "status": "running",
        "model": "SAM 2.1 tiny (loads on first request)"
    }


# =====================================
# AI WRIST SEGMENTATION
# =====================================

@app.post("/refine-fit")
async def refine_fit(
    wrist_image: UploadFile = File(...),
    geometry: str = Form(...)
):
    if wrist_image.content_type not in (
        "image/jpeg",
        "image/png"
    ):
        raise HTTPException(
            status_code=400,
            detail="Use JPG or PNG wrist photo"
        )

    raw = await wrist_image.read(6_000_001)

    if len(raw) > 6_000_000:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds 6 MB"
        )

    try:
        image = ImageOps.exif_transpose(
            Image.open(io.BytesIO(raw))
        ).convert("RGB")

    except (
        UnidentifiedImageError,
        OSError
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        ) from exc

    width, height = image.size

    if width * height > 3_000_000:
        raise HTTPException(
            status_code=413,
            detail="Reduce image below 3 megapixels"
        )

    wrist, axis, palm_width = parse_geometry(
        geometry,
        width,
        height
    )

    magnitude = float(np.linalg.norm(axis))

    if magnitude < 0.1:
        raise HTTPException(
            status_code=400,
            detail="Invalid wrist direction"
        )

    d = axis / magnitude

    positive = [
        wrist + d * palm_width * t
        for t in (0.22, 0.55, 0.85)
    ]

    point_coords = np.array(
        positive,
        dtype=np.float32
    )

    point_labels = np.ones(
        len(positive),
        dtype=np.int64
    )

    if any(
        not (
            0 <= p[0] < width
            and 0 <= p[1] < height
        )
        for p in positive
    ):
        return {
            "usedAI": False,
            "reason": "Forearm extends beyond photo"
        }

    try:
        torch, processor, model, device = get_model()

        inputs = processor(
            images=image,
            input_points=[[point_coords.tolist()]],
            input_labels=[[point_labels.tolist()]],
            return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)

        candidates = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"]
        )[0][0]

        scores = (
            outputs.iou_scores
            .detach()
            .cpu()
            .reshape(-1)
            .numpy()
        )

        masks = candidates.detach().cpu().numpy()

    except Exception as exc:
        print("SAM 2 error:", repr(exc))

        raise HTTPException(
            status_code=503,
            detail="SAM 2 model not ready; check server logs"
        ) from exc

    best = None
    best_score = -1.0

    for mask, score in zip(masks, scores):
        mask = np.asarray(mask) > 0.5

        fraction = float(mask.mean())

        if fraction < 0.015 or fraction > 0.72:
            continue

        if not all(
            mask[
                int(round(p[1])),
                int(round(p[0]))
            ]
            for p in positive
        ):
            continue

        result = geometry_from_mask(
            mask,
            wrist,
            d,
            palm_width
        )

        if (
            result is not None
            and float(score) > best_score
        ):
            best = result
            best_score = float(score)

    if best is None or best_score < 0.70:
        return {
            "usedAI": False,
            "reason": "Unreliable wrist segmentation"
        }

    return {
        "usedAI": True,
        "qualityEstimate": round(best_score, 3),
        **best,
    }


# =====================================
# AUTOMATIC WATCH BACKGROUND REMOVAL
# =====================================

@lru_cache(maxsize=1)
def get_watch_removal_session():
    from rembg import new_session

    return new_session("isnet-general-use")


@app.post("/remove-watch-background")
async def remove_watch_background(
    watch_image: UploadFile = File(...)
):
    if watch_image.content_type not in (
        "image/jpeg",
        "image/png",
        "image/webp"
    ):
        raise HTTPException(
            status_code=400,
            detail="Use JPG, PNG or WebP watch images"
        )

    raw = await watch_image.read(8_000_001)

    if len(raw) > 8_000_000:
        raise HTTPException(
            status_code=413,
            detail="Watch image exceeds 8 MB"
        )

    try:
        image = ImageOps.exif_transpose(
            Image.open(io.BytesIO(raw))
        )

        if image.width * image.height > 8_000_000:
            raise HTTPException(
                status_code=413,
                detail="Watch image exceeds 8 megapixels"
            )

        image = image.convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid watch image"
        ) from exc

    try:
        from rembg import remove

        output = remove(
            image,
            session=get_watch_removal_session()
        ).convert("RGBA")

    except Exception as exc:
        print(
            "Watch background removal error:",
            repr(exc)
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Background removal unavailable. "
                "Check backend logs and rembg installation."
            )
        ) from exc

    alpha = np.asarray(
        output.getchannel("A")
    )

    if float(np.mean(alpha < 245)) < 0.01:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not isolate the watch. "
                "Try a plain, contrasting background."
            )
        )

    if float(np.mean(alpha > 50)) < 0.01:
        raise HTTPException(
            status_code=422,
            detail=(
                "Watch was not detected. "
                "Try a clearer watch photo."
            )
        )

    buffer = io.BytesIO()

    output.save(
        buffer,
        format="PNG"
    )

    return Response(
        content=buffer.getvalue(),
        media_type="image/png"
    )
