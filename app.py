"""
Gradio web-interface for the Super-Resolution application.

Usage
-----
    python app.py                        # demo mode (random weights)
    python app.py --checkpoint_x2 checkpoints/best_x2.pth
    python app.py --checkpoint_x2 checkpoints/best_x2.pth \\
                  --checkpoint_x4 checkpoints/best_x4.pth

Opens a local browser tab with:
  - image upload
  - scale factor selector (x2 / x4)
  - side-by-side comparison (original | SR result)
  - download button for the output
  - optional quality metrics if a reference HR image is provided
"""

import argparse
import os
import time
import tempfile

import gradio as gr
import numpy as np
import torch
from PIL import Image

from models.srcnn import get_model
from utils.image_utils import (
    numpy_to_tensor,
    tensor_to_numpy,
    resize_bicubic,
)
from utils.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Global model cache (loaded once per scale factor)
# ---------------------------------------------------------------------------
_models: dict = {}


def _load_model(scale: int, checkpoint: str | None, device: str) -> torch.nn.Module:
    """Load or return cached model for the given scale factor."""
    key = (scale, checkpoint)
    if key not in _models:
        model = get_model(scale_factor=scale, device=device)
        if checkpoint and os.path.isfile(checkpoint):
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state)
            print(f"[INFO] Loaded checkpoint for x{scale}: {checkpoint}")
        else:
            print(f"[WARN] No checkpoint for x{scale} — using untrained weights")
        model.eval()
        _models[key] = model
    return _models[key]


# ---------------------------------------------------------------------------
# Core processing function called by Gradio
# ---------------------------------------------------------------------------
def process_image(
    input_image: np.ndarray | None,
    scale_factor: str,
    checkpoint_x2: str | None = None,
    checkpoint_x4: str | None = None,
):
    """Run super-resolution and return results for the Gradio interface."""
    if input_image is None:
        raise gr.Error("Пожалуйста, загрузите изображение / Please upload an image")

    scale = int(scale_factor.replace("x", ""))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pick checkpoint
    ckpt = checkpoint_x2 if scale == 2 else checkpoint_x4

    model = _load_model(scale, ckpt, device)

    # Prepare tensor
    if input_image.dtype != np.uint8:
        input_image = (input_image * 255).astype(np.uint8) if input_image.max() <= 1.0 else input_image.astype(np.uint8)

    lr_tensor = numpy_to_tensor(input_image).to(device)

    # Inference
    with torch.no_grad():
        t0 = time.time()
        sr_tensor = model(lr_tensor)
        elapsed = time.time() - t0

    sr_image = tensor_to_numpy(sr_tensor)

    # Bicubic baseline for comparison
    bicubic_image = resize_bicubic(input_image, scale)

    # Metrics (SR vs bicubic — not ground-truth, but useful for demonstration)
    h_min = min(sr_image.shape[0], bicubic_image.shape[0])
    w_min = min(sr_image.shape[1], bicubic_image.shape[1])
    metrics = compute_metrics(
        sr_image[:h_min, :w_min], bicubic_image[:h_min, :w_min]
    )

    h_in, w_in = input_image.shape[:2]
    h_out, w_out = sr_image.shape[:2]

    info_text = (
        f"Масштаб: x{scale}\n"
        f"Вход: {w_in}×{h_in} → Выход: {w_out}×{h_out}\n"
        f"Время инференса: {elapsed:.3f} сек\n"
        f"PSNR (SR vs Bicubic): {metrics['psnr']} дБ\n"
        f"SSIM (SR vs Bicubic): {metrics['ssim']}"
    )

    # Save to temp file for download
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(sr_image).save(tmp.name)

    return sr_image, bicubic_image, info_text, tmp.name


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------
def build_app(checkpoint_x2=None, checkpoint_x4=None):
    with gr.Blocks(
        title="Super-Resolution App",
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as demo:
        gr.Markdown(
            """
            # 🔬 Нейросетевое суперразрешение изображений
            ### ESPCN Super-Resolution Application

            Загрузите изображение низкого разрешения, выберите коэффициент
            увеличения и получите результат суперразрешения с помощью нейросети.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(
                    label="Входное изображение (LR)",
                    type="numpy",
                    sources=["upload", "clipboard"],
                )
                scale_radio = gr.Radio(
                    choices=["x2", "x4"],
                    value="x2",
                    label="Коэффициент увеличения",
                )
                run_btn = gr.Button("▶ Запустить суперразрешение", variant="primary")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Результат SR"):
                        output_sr = gr.Image(label="Super-Resolution", type="numpy")
                    with gr.Tab("Сравнение с Bicubic"):
                        output_bic = gr.Image(label="Bicubic (baseline)", type="numpy")

                info_box = gr.Textbox(
                    label="Информация",
                    lines=5,
                    interactive=False,
                )
                download_btn = gr.File(label="Скачать результат")

        # Wire up the button
        run_btn.click(
            fn=lambda img, sc: process_image(img, sc, checkpoint_x2, checkpoint_x4),
            inputs=[input_img, scale_radio],
            outputs=[output_sr, output_bic, info_box, download_btn],
        )

        gr.Markdown(
            """
            ---
            **Архитектура модели:** ESPCN (Efficient Sub-Pixel CNN)  
            **Стек:** PyTorch · OpenCV · Gradio  
            """
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_x2", type=str, default=None)
    parser.add_argument("--checkpoint_x4", type=str, default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    # Try default paths if not specified
    if args.checkpoint_x2 is None:
        for p in ("checkpoints/best_x2.pth", "checkpoints/espcn_x2_final.pth"):
            if os.path.isfile(p):
                args.checkpoint_x2 = p
                break
    if args.checkpoint_x4 is None:
        for p in ("checkpoints/best_x4.pth", "checkpoints/espcn_x4_final.pth"):
            if os.path.isfile(p):
                args.checkpoint_x4 = p
                break

    demo = build_app(args.checkpoint_x2, args.checkpoint_x4)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
