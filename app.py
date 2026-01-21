# --- Fix Windows pickle paths BEFORE anything else ---
import pathlib
pathlib.WindowsPath = pathlib.PosixPath

import io
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
from PIL import Image, ImageOps, UnidentifiedImageError
from fastai.vision.all import PILImage
import torch

_learner = None

try:
    import pillow_heif
except ImportError:
    pillow_heif = None


def load_model():
    global _learner
    if _learner is None:
        _learner = torch.load("vehicle_model.pkl", map_location="cpu")
    return _learner


def _ensure_heif():
    if pillow_heif is None:
        raise gr.Error(
            "HEIC/HEIF support is disabled. Install pillow-heif to enable it."
        )
    pillow_heif.register_heif_opener()


def _describe_image(img: Image.Image) -> Dict[str, Any]:
    return {
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
    }


def _metadata_markdown(before: Image.Image, after: Image.Image) -> str:
    return "\n".join(
        [
            f"- Original: {before.width}x{before.height} ({before.mode})",
            f"- Processed: {after.width}x{after.height} ({after.mode})",
        ]
    )


def _load_pil_image(
    file: Union[str, Image.Image, Any], enable_heif: bool = False
) -> Image.Image:
    if isinstance(file, Image.Image):
        return ImageOps.exif_transpose(file).convert("RGB")

    file_path = file if isinstance(file, str) else getattr(file, "name", None)
    if not file_path:
        raise gr.Error("Could not read the uploaded image.")

    if enable_heif:
        _ensure_heif()

    with open(file_path, "rb") as f:
        raw = f.read()

    print("Uploaded file:", file_path)
    print("File size:", len(raw))
    print("First bytes:", raw[:16])

    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img
    except UnidentifiedImageError:
        raise gr.Error(
            "Unsupported image format. Try a standard JPG or PNG or enable HEIC support."
        )


def _preprocess_image(
    img: Image.Image,
    resize_longest: Optional[int] = None,
    auto_contrast: bool = False,
) -> Image.Image:
    work = img.copy()
    if resize_longest:
        work = ImageOps.contain(work, (resize_longest, resize_longest))
    if auto_contrast:
        work = ImageOps.autocontrast(work)
    return work


def _predict(
    file: Union[str, Image.Image, Any],
    resize_longest: Optional[int] = None,
    enable_heif: bool = False,
    auto_contrast: bool = False,
):
    learner = load_model()

    pil_img = _load_pil_image(file, enable_heif=enable_heif)
    before_meta = _describe_image(pil_img)

    processed = _preprocess_image(
        pil_img,
        resize_longest=resize_longest,
        auto_contrast=auto_contrast,
    )
    after_meta = _describe_image(processed)

    fa_img = PILImage.create(processed)
    _, _, probs = learner.predict(fa_img)

    vocab = list(learner.dls.vocab) if hasattr(learner, "dls") else None
    top_probs, top_idxs = probs.topk(5)
    top_probs = top_probs.tolist()
    top_idxs = top_idxs.tolist()

    if vocab:
        mapping = {vocab[int(i)]: float(p) for p, i in zip(top_probs, top_idxs)}
    else:
        mapping = {f"class_{int(i)}": float(p) for p, i in zip(top_probs, top_idxs)}

    top_label = max(mapping.items(), key=lambda kv: kv[1])[0]

    meta_md = _metadata_markdown(
        Image.new("RGB", (before_meta["width"], before_meta["height"])),
        Image.new("RGB", (after_meta["width"], after_meta["height"])),
    )

    meta_md = "\n".join(
        [
            f"- Original: {before_meta['width']}x{before_meta['height']} ({before_meta['mode']})",
            f"- Processed: {after_meta['width']}x{after_meta['height']} ({after_meta['mode']})",
        ]
    )

    return top_label, mapping, processed, meta_md


def _predict_batch(
    files: List[Any],
    resize_longest: Optional[int] = None,
    enable_heif: bool = False,
    auto_contrast: bool = False,
):
    if not files:
        return []

    rows = []
    for file in files:
        try:
            top_label, mapping, _, _ = _predict(
                file,
                resize_longest=resize_longest,
                enable_heif=enable_heif,
                auto_contrast=auto_contrast,
            )
            top_prob = max(mapping.values()) if mapping else 0.0
            rows.append(
                {
                    "file": getattr(file, "name", None) or str(file),
                    "top_label": top_label,
                    "top_probability": round(float(top_prob), 4),
                    "top5": mapping,
                }
            )
        except Exception as ex:  # keep batch running even if one fails
            rows.append(
                {
                    "file": getattr(file, "name", None) or str(file),
                    "top_label": "error",
                    "top_probability": "-",
                    "top5": str(ex),
                }
            )
    return rows


def _class_guide() -> str:
    vocab = []
    try:
        vocab = list(load_model().dls.vocab)
    except Exception:
        pass

    if not vocab:
        return "Model vocab unavailable. Upload an image to load the model first."

    lines = ["| Class | Description |", "| --- | --- |"]
    for cls in vocab:
        lines.append(f"| {cls} | Typical appearance of {cls} vehicles. |")
    return "\n".join(lines)


example_images = [
    "https://upload.wikimedia.org/wikipedia/commons/3/3e/Tesla_Model_3_parked%2C_front_driver_side.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/9/98/2018_Toyota_Camry_SE_front_3.10.18.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/e/e0/2019_Ford_Ranger_Wildtrak_2.0.jpg",
]


with gr.Blocks(title="🚗 Vehicle Classification") as demo:
    gr.Markdown(
        "# Vehicle Classification\nUpload a vehicle image from file, drag-and-drop, or the camera."
    )

    with gr.Row():
        with gr.Column():
            image_file = gr.File(
                label="Upload image (drag-and-drop)",
                file_types=["image"],
                file_count="single",
            )
            webcam_input = gr.Image(
                label="Capture from camera",
                source="webcam",
                type="pil",
            )

            gr.Markdown("**Examples**")
            gr.Examples(examples=example_images, inputs=image_file)

            with gr.Accordion("Advanced", open=False):
                resize_longest = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=1024,
                    step=64,
                    label="Resize longest edge (px)",
                    info="Applied before prediction to keep sizes manageable.",
                )
                auto_contrast = gr.Checkbox(
                    value=True, label="Auto-contrast", info="Improves dull lighting."
                )
                enable_heif = gr.Checkbox(
                    value=False,
                    label="Enable HEIC/HEIF support",
                    info="Requires pillow-heif; turn on only if you need it.",
                )

            with gr.Accordion("Class guide", open=False):
                class_md = gr.Markdown(_class_guide())

        with gr.Column():
            top_label = gr.Textbox(label="Top prediction", interactive=False)
            top_probs = gr.Label(num_top_classes=5, label="Top 5 probabilities")
            preview = gr.Image(label="What it saw", type="pil")
            meta = gr.Markdown(label="Image details")

    with gr.Tab("Batch mode"):
        batch_files = gr.File(
            label="Upload multiple images", file_types=["image"], file_count="multiple"
        )
        run_batch = gr.Button("Run batch")
        batch_table = gr.Dataframe(
            headers=["file", "top_label", "top_probability", "top5"],
            interactive=False,
        )

    def _update_class_guide():
        return _class_guide()

    def _predict_single_wrapper(file, resize_longest, enable_heif, auto_contrast):
        return _predict(
            file,
            resize_longest=resize_longest,
            enable_heif=enable_heif,
            auto_contrast=auto_contrast,
        )

    image_file.change(
        _predict_single_wrapper,
        inputs=[image_file, resize_longest, enable_heif, auto_contrast],
        outputs=[top_label, top_probs, preview, meta],
    )

    webcam_input.change(
        _predict_single_wrapper,
        inputs=[webcam_input, resize_longest, enable_heif, auto_contrast],
        outputs=[top_label, top_probs, preview, meta],
    )

    run_batch.click(
        _predict_batch,
        inputs=[batch_files, resize_longest, enable_heif, auto_contrast],
        outputs=batch_table,
    )

    image_file.change(_update_class_guide, inputs=None, outputs=class_md)
    webcam_input.change(_update_class_guide, inputs=None, outputs=class_md)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
