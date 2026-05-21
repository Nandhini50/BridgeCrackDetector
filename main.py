import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
IMAGE_PATH  = "bridge.jpg"        # place bridge.jpg next to this script
OUTPUT_PATH = "crack_detection_result.png"

# ─────────────────────────────────────────────
#  STAGE FUNCTIONS
# ─────────────────────────────────────────────

def load_original(path: str) -> np.ndarray:
    """Load image in BGR → convert to RGB for display."""
    img = cv2.imread(path)
    if img is None: 
        print(f"[ERROR] Could not read '{path}'.")
        print("  Make sure bridge.jpg is in the same folder as main.py.")
        sys.exit(1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def stage1_grayscale_denoise(rgb: np.ndarray) -> np.ndarray:
    """
    Stage 1 – Grayscale + Noise Removal
      • Convert to grayscale
      • Non-Local Means denoising  ← preserves edges better than Gaussian
    """
    gray     = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=9,
                                        templateWindowSize=7,
                                        searchWindowSize=21)
    return denoised


def stage2_enhance(gray: np.ndarray) -> np.ndarray:
    """
    Stage 2 – Contrast Enhancement
      • CLAHE  ← boosts local contrast without blowing highlights
      • Unsharp mask ← sharpens fine crack edges
    """
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blur  = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
    sharp = cv2.addWeighted(enhanced, 1.6, blur, -0.6, 0)
    return sharp


def stage3_threshold(enhanced: np.ndarray) -> np.ndarray:
    """
    Stage 3 – Adaptive Thresholding
      • Gaussian adaptive threshold handles uneven concrete illumination
      • Morphological open+close removes speckles & closes crack gaps
    """
    thresh = cv2.adaptiveThreshold(
        enhanced,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=4
    )
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh,   cv2.MORPH_OPEN,  kernel, iterations=1)
    closed  = cv2.morphologyEx(cleaned,  cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def stage4_crack_detection(original_rgb: np.ndarray,
                            thresh: np.ndarray) -> np.ndarray:
    """
    Stage 4 – Crack Detection & Highlighted Overlay
      • Dilate mask for visibility
      • Filter contours by area (remove noise)
      • Semi-transparent red fill on crack regions
      • Neon-green contour outlines + cyan bounding boxes + labels
      • Stats banner at top
    """
    k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask     = cv2.dilate(thresh, k_dilate, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    min_area = 50
    cracks   = [c for c in contours if cv2.contourArea(c) >= min_area]

    result  = original_rgb.copy()

    # Semi-transparent red fill
    overlay = result.copy()
    cv2.drawContours(overlay, cracks, -1, (220, 30, 30), thickness=cv2.FILLED)
    cv2.addWeighted(overlay, 0.45, result, 0.55, 0, result)

    # Neon-green contour outlines
    cv2.drawContours(result, cracks, -1, (0, 255, 80), thickness=1)

    # Cyan bounding boxes + labels
    for i, cnt in enumerate(cracks, 1):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 230, 255), thickness=1)
        cv2.putText(result, f"C{i}", (x, max(y - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 0), 1, cv2.LINE_AA)

    # Stats banner
    total_px  = int(np.sum(mask > 0))
    total_img = mask.shape[0] * mask.shape[1]
    pct       = total_px / total_img * 100

    cv2.rectangle(result, (0, 0), (result.shape[1], 28), (20, 20, 20), cv2.FILLED)
    cv2.putText(result,
                f"Cracks detected: {len(cracks)}  |  "
                f"Affected area: {pct:.2f}%  |  "
                f"Crack pixels: {total_px:,}",
                (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (0, 255, 180), 1, cv2.LINE_AA)

    return result


# ─────────────────────────────────────────────
#  PANEL COMPOSER
# ─────────────────────────────────────────────

def build_panel(original, gray_denoised, enhanced, thresh, detection):
    """5-panel figure showing every processing stage."""
    fig = plt.figure(figsize=(22, 9), facecolor="#0d0d0d")
    fig.suptitle(
        "BRIDGE CRACK DETECTION  —  IMAGE PROCESSING PIPELINE",
        fontsize=15, fontweight="bold",
        color="#e8e8e8", y=0.97, fontfamily="monospace"
    )

    gs = gridspec.GridSpec(1, 5, figure=fig,
                           left=0.02, right=0.98,
                           top=0.88, bottom=0.06,
                           wspace=0.04)

    stages = [
        (original,      "ORIGINAL",               "color"),
        (gray_denoised, "GRAYSCALE + DENOISED",    "gray"),
        (enhanced,      "CLAHE + SHARPENED",       "gray"),
        (thresh,        "ADAPTIVE THRESHOLD",      "gray"),
        (detection,     "CRACK DETECTION OVERLAY", "color"),
    ]
    colors = ["#60c8ff", "#a8ff78", "#ffd86e", "#ff8c42", "#ff4d6d"]

    for col, (img, title, mode) in enumerate(stages):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img, cmap=("gray" if mode == "gray" else None))
        ax.set_title(title, color=colors[col],
                     fontsize=8.5, fontweight="bold",
                     fontfamily="monospace", pad=6)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(colors[col])
            spine.set_linewidth(1.2)
            spine.set_visible(True)
        ax.text(0.03, 0.97, f"STAGE {col + 1}",
                transform=ax.transAxes,
                color=colors[col], fontsize=7, fontweight="bold",
                fontfamily="monospace", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0d0d0d",
                          alpha=0.7, edgecolor=colors[col], linewidth=0.8))
    return fig


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run():
    print("=" * 55)
    print("  BRIDGE CRACK DETECTION PIPELINE")
    print("=" * 55)

    print(f"[1/5] Loading '{IMAGE_PATH}' …")
    original = load_original(IMAGE_PATH)
    h, w = original.shape[:2]
    print(f"      Image: {w} × {h} px")

    print("[2/5] Stage 1 – Grayscale + Noise Removal …")
    gray_denoised = stage1_grayscale_denoise(original)

    print("[3/5] Stage 2 – CLAHE + Unsharp Enhancement …")
    enhanced = stage2_enhance(gray_denoised)

    print("[4/5] Stage 3 – Adaptive Thresholding …")
    thresh = stage3_threshold(enhanced)

    print("[5/5] Stage 4 – Crack Detection & Overlay …")
    detection = stage4_crack_detection(original, thresh)

    print("\n  Composing panel figure …")
    fig = build_panel(original, gray_denoised, enhanced, thresh, detection)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Saved → '{OUTPUT_PATH}'")

    plt.show()
    print("\n  Done. Close the window to exit.")


if __name__ == "__main__":
    run()
