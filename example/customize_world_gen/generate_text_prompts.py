"""Generate one Qwen3-VL scene prompt per approved camera turn."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from camera_trajectory import parse_camera_movement
from prompt_policy import Qwen3VLPromptPlanner, load_dl3dv_prompt_examples


def generate_prompt_file(
    *,
    image_path: Path,
    movement_path: Path,
    output_path: Path,
    dl3dv_source: Path,
    model_dir: str,
    device: str,
    dtype: str,
    attn_impl: str,
    max_new_tokens: int,
    max_attempts: int,
) -> list[str]:
    """Run Qwen and persist only the final line-oriented prompt file."""

    commands = parse_camera_movement(movement_path)
    style_examples = load_dl3dv_prompt_examples(dl3dv_source)
    planner = Qwen3VLPromptPlanner(
        model_dir,
        device=device,
        dtype=dtype,
        attn_impl=attn_impl,
        max_new_tokens=max_new_tokens,
    )
    with Image.open(image_path) as image:
        prompts = planner.generate(
            image.convert("RGB"),
            commands,
            style_examples,
            max_attempts=max_attempts,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--camera-movement", type=Path, required=True)
    parser.add_argument("--output-prompt", type=Path, required=True)
    parser.add_argument(
        "--dl3dv-source",
        type=Path,
        required=True,
        help="DL3DV wan_standard root used for prompt style examples",
    )
    parser.add_argument(
        "--model-dir", default="model_weights/Qwen3-VL-8B-Instruct"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--attn-impl",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="flash_attention_2",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"image not found: {args.image}")
    if not args.camera_movement.is_file():
        raise FileNotFoundError(
            f"camera movement not found: {args.camera_movement}"
        )
    prompts = generate_prompt_file(
        image_path=args.image.resolve(),
        movement_path=args.camera_movement.resolve(),
        output_path=args.output_prompt.expanduser().resolve(),
        dl3dv_source=args.dl3dv_source.expanduser().resolve(),
        model_dir=args.model_dir,
        device=args.device,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        max_new_tokens=args.max_new_tokens,
        max_attempts=args.max_attempts,
    )
    print(f"Wrote {len(prompts)} prompts: {args.output_prompt.expanduser().resolve()}")


if __name__ == "__main__":
    main()