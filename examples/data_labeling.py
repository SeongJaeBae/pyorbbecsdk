import cv2
import time
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# =========================
# Config
# =========================
BASE_DIR = Path("/media/nvidia/T9/합포장 과제/nas_data")

# COLOR_DIR = BASE_DIR / "color"
COLOR_DIR = BASE_DIR / "Images"
NEW_PRED_DIR = BASE_DIR / "new_pred_obb"
NEW_LABEL_DIR = BASE_DIR / "new_label_obb"

NEW_PRED_DIR.mkdir(parents=True, exist_ok=True)
NEW_LABEL_DIR.mkdir(parents=True, exist_ok=True)

SCORE_THRESH = 0.10
MASK_THRESH = 0.4
MIN_AREA = 100

# CONCEPTS = ["mealkit packet"]
    
# 
# CONCEPTS = ["food packet"]
CONCEPTS = ["package"]
# 

USE_AUTOCAST = True
AUTOCAST_DTYPE = torch.bfloat16


# =========================
# Utils
# =========================
def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if torch.is_tensor(x):
        if x.dtype == torch.bfloat16:
            x = x.float()
        return x.numpy()
    return np.array(x)


def run_sam3(processor, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    if torch.cuda.is_available() and USE_AUTOCAST:
        with torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
            inference_state = processor.set_image(pil_image)
    else:
        inference_state = processor.set_image(pil_image)

    all_masks = []
    all_scores = []

    for concept in CONCEPTS:
        if torch.cuda.is_available() and USE_AUTOCAST:
            with torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
                output = processor.set_text_prompt(
                    state=inference_state,
                    prompt=concept
                )
        else:
            output = processor.set_text_prompt(
                state=inference_state,
                prompt=concept
            )

        masks = output.get("masks", [])
        scores = output.get("scores", [])

        scores = to_numpy(scores)

        if masks is None or len(masks) == 0:
            continue

        if scores is None or len(scores) == 0:
            scores = np.ones((len(masks),), dtype=np.float32)

        for i in range(len(masks)):
            if float(scores[i]) >= SCORE_THRESH:
                all_masks.append(masks[i])
                all_scores.append(float(scores[i]))

    return all_masks, all_scores


def mask_to_obb_points(mask, image_w, image_h):
    mask_np = to_numpy(mask)
    if mask_np is None:
        return None

    mask_np = np.squeeze(mask_np)
    if mask_np.ndim != 2:
        return None

    mask_bin = (mask_np > MASK_THRESH).astype(np.uint8)

    if mask_bin.shape[0] != image_h or mask_bin.shape[1] != image_w:
        mask_bin = cv2.resize(
            mask_bin,
            (image_w, image_h),
            interpolation=cv2.INTER_NEAREST
        )

    contours, _ = cv2.findContours(
        mask_bin * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)

    if area < MIN_AREA:
        return None

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    return box


def normalize_obb_points(box, image_w, image_h):
    norm = []

    for x, y in box:
        x = np.clip(x / image_w, 0.0, 1.0)
        y = np.clip(y / image_h, 0.0, 1.0)
        norm.extend([x, y])

    return norm


def draw_obb(frame, box, score=None):
    vis = frame.copy()

    cv2.polylines(
        vis,
        [box],
        isClosed=True,
        color=(0, 0, 255),
        thickness=2
    )

    for i, (x, y) in enumerate(box):
        cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.putText(
            vis,
            str(i + 1),
            (int(x), int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    if score is not None:
        x, y = box[0]
        cv2.putText(
            vis,
            f"{score:.2f}",
            (int(x), max(0, int(y) - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    return vis


def draw_all_obbs(frame, obb_list, score_list):
    vis = frame.copy()

    for box, score in zip(obb_list, score_list):
        cv2.polylines(
            vis,
            [box],
            isClosed=True,
            color=(0, 0, 255),
            thickness=2
        )

        for i, (x, y) in enumerate(box):
            cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)

        x, y = box[0]
        cv2.putText(
            vis,
            f"{score:.2f}",
            (int(x), max(0, int(y) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    return vis


# =========================
# Main
# =========================
def main():
    print("[INFO] Loading SAM3 model...")
    t0 = time.time()

    sam3_model = build_sam3_image_model()

    if torch.cuda.is_available():
        sam3_model = sam3_model.cuda()

    sam3_model.eval()
    processor = Sam3Processor(sam3_model)

    print(f"[INFO] SAM3 loaded: {time.time() - t0:.2f}s")

    # color_files = sorted(COLOR_DIR.glob("color_*.jpg"))
    color_files = sorted(COLOR_DIR.glob("data_*.jpg"))

    print(f"[INFO] color files: {len(color_files)}")
    print(f"[INFO] pred output: {NEW_PRED_DIR}")
    print(f"[INFO] label output: {NEW_LABEL_DIR}")

    with torch.inference_mode():
        for idx, color_path in enumerate(color_files, start=1):
            frame = cv2.imread(str(color_path))

            if frame is None:
                print(f"[SKIP] cannot read: {color_path}")
                continue

            h, w = frame.shape[:2]

            timestamp = color_path.stem.replace("color_", "")

            pred_path = NEW_PRED_DIR / f"pred_{timestamp}.jpg"
            label_path = NEW_LABEL_DIR / f"label_{timestamp}.txt"

            masks, scores = run_sam3(processor, frame)

            obb_list = []
            obb_scores = []
            label_lines = []

            for mask, score in zip(masks, scores):
                box = mask_to_obb_points(mask, w, h)

                if box is None:
                    continue

                obb_list.append(box)
                obb_scores.append(score)

                norm_points = normalize_obb_points(box, w, h)

                line = "0 " + " ".join([f"{v:.6f}" for v in norm_points])
                label_lines.append(line)

            pred_vis = draw_all_obbs(frame, obb_list, obb_scores)

            cv2.imwrite(str(pred_path), pred_vis)

            with open(label_path, "w", encoding="utf-8") as f:
                if label_lines:
                    f.write("\n".join(label_lines) + "\n")
                else:
                    f.write("")

            print(
                f"[{idx}/{len(color_files)}] "
                f"SAVE pred={pred_path.name}, label={label_path.name} "
                f"| objects={len(obb_list)}"
            )

    print("[DONE] OBB prediction + label 저장 완료")


if __name__ == "__main__":
    main()