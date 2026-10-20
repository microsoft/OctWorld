"""Qwen3-VL policy for one conservative scene prompt per camera turn."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from PIL import Image

from camera_trajectory import Command, resolve_commands


SYSTEM_PROMPT = (
    "You write concise English scene prompts for navigation-oriented video "
    "generation. Ground predictions in the initial image, preserve scene "
    "identity, and prefer conservative local continuation over invention."
)

ACTION_EFFECTS = {
    "move_left": "Reveal only the immediately adjacent area on the left.",
    "move_right": "Reveal only the immediately adjacent area on the right.",
    "push_in": "Show closer detail of already visible central content.",
    "pull_out": "Show a modestly wider context around the visible scene.",
    "pan_left": "Shift attention to the nearby left side while staying in place.",
    "pan_right": "Shift attention to the nearby right side while staying in place.",
    "orbit_left": "Reveal a plausible nearby left-facing side of visible objects.",
    "orbit_right": "Reveal a plausible nearby right-facing side of visible objects.",
    "pull_left": "Show a wider view and the immediately adjacent left area.",
    "pull_right": "Show a wider view and the immediately adjacent right area.",
    "push_left": "Move closer while revealing only the nearby left side.",
    "push_right": "Move closer while revealing only the nearby right side.",
}


def load_dl3dv_prompt_examples(
    source_dir: str | Path, max_examples: int = 8
) -> list[str]:
    """Load deterministic style examples from a DL3DV standard dataset."""

    source_dir = Path(source_dir).expanduser().resolve()
    instances_path = source_dir / "instances.json"
    if not instances_path.is_file():
        raise FileNotFoundError(f"DL3DV instances file not found: {instances_path}")
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    if not isinstance(instances, list) or not instances:
        raise ValueError(f"instances must be a non-empty list: {instances_path}")

    sample_count = min(max_examples, len(instances))
    sample_indices = [
        index * len(instances) // sample_count for index in range(sample_count)
    ]
    examples = []
    for index in sample_indices:
        prompt_relpath = instances[index].get("prompt_file")
        if not prompt_relpath:
            raise ValueError(f"DL3DV instance {index} has no prompt_file")
        prompt_path = source_dir / prompt_relpath
        lines = [
            " ".join(line.split())
            for line in prompt_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            raise ValueError(f"DL3DV prompt file is empty: {prompt_path}")
        examples.append(lines[len(lines) // 2])
    return examples


def build_batch_instruction(
    commands: Sequence[Command], style_examples: Sequence[str]
) -> str:
    """Build an image-grounded request without numeric movement parameters."""

    resolved = resolve_commands(commands)
    if not style_examples:
        raise ValueError("at least one DL3DV style example is required")
    turns = "\n".join(
        f"{index + 1}. {action}: {ACTION_EFFECTS[action]}"
        for index, (action, _) in enumerate(resolved)
    )
    examples = "\n".join(f'- "{example}"' for example in style_examples)
    schema = json.dumps(
        {"prompts": [f"prompt {index + 1}" for index in range(len(resolved))]}
    )
    return f"""The attached image is the only observed frame. The target video will navigate through {len(resolved)} consecutive camera turns.

Ordered camera turns:
{turns}

Write exactly one English scene prompt for each turn in order.

Rules:
1. Preserve the same location, architecture, objects, materials, lighting, weather, and mood.
2. Infer only a small locally adjacent continuation implied by each turn. Do not invent a new room, distant area, person, animal, event, sign, brand, or unsupported object.
3. Treat later turns cumulatively. A reversing turn may return toward an earlier composition.
4. Prefer repeating or reframing known content when the unseen continuation is uncertain.
5. Focus on scene content. Brief view or composition wording is acceptable, but avoid lengthy camera-operation narration.
6. Match the examples: one concise, self-contained, present-tense sentence per turn.
7. Return valid JSON only in this exact schema: {schema}

Representative source DL3DV prompts:
{examples}
"""


def build_turn_instruction(
    commands: Sequence[Command],
    turn_index: int,
    previous_prompts: Sequence[str],
    style_examples: Sequence[str],
) -> str:
    """Build one fallback request using action names but no numeric arguments."""

    resolved = resolve_commands(commands)
    if not 0 <= turn_index < len(resolved):
        raise IndexError(f"turn index out of range: {turn_index}")
    action = resolved[turn_index][0]
    action_history = " -> ".join(item[0] for item in resolved[: turn_index + 1])
    previous = (
        "\n".join(f'- "{prompt}"' for prompt in previous_prompts)
        if previous_prompts
        else "(none; this is the first turn)"
    )
    examples = "\n".join(f'- "{example}"' for example in style_examples)
    return f"""The attached image is the only observed frame. Write the scene prompt for turn {turn_index + 1}.

Action history: {action_history}
Current action: {action}
Current action meaning: {ACTION_EFFECTS[action]}

Earlier prompts:
{previous}

Keep the same place and visual conditions. Mention only visible or strongly implied content, and continue cumulatively from earlier turns. Prefer reframing known content over inventing a person, animal, object, room, event, or distant area.

Use one concise present-tense English sentence in the style of these source DL3DV prompts:
{examples}

Return valid JSON only: {{"prompts": ["prompt"]}}.
"""


def _extract_json(response: str) -> object:
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Qwen response does not contain a JSON object") from None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in Qwen response: {error}") from error


def parse_prompt_response(response: str, expected_count: int) -> list[str]:
    """Enforce the structural prompt contract required by world generation."""

    payload = _extract_json(response)
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    if not isinstance(prompts, list):
        raise ValueError("Qwen response must contain a prompts list")
    if len(prompts) != expected_count:
        raise ValueError(
            f"expected {expected_count} prompts, received {len(prompts)}"
        )
    normalized = []
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, str):
            raise ValueError(f"prompt {index} is not a string")
        prompt = " ".join(prompt.split()).strip().strip('"')
        if not prompt:
            raise ValueError(f"prompt {index} is empty")
        normalized.append(prompt)
    return normalized


class Qwen3VLPromptPlanner:
    """Local deterministic Qwen3-VL planner for navigation prompts."""

    def __init__(
        self,
        model_dir: str,
        *,
        device: str = "cuda",
        dtype: str = "bf16",
        attn_impl: str = "flash_attention_2",
        max_new_tokens: int = 1024,
        max_pixels: int = 1280 * 32 * 32,
        min_pixels: int = 256 * 32 * 32,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.processor.image_processor.size = {
            "longest_edge": max_pixels,
            "shortest_edge": min_pixels,
        }
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            dtype=dtype_map[dtype],
            attn_implementation=attn_impl,
            device_map=device,
        ).eval()

    def _generate_once(self, image: Image.Image, instruction: str) -> str:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = generated[0, inputs.input_ids.shape[1] :]
        return self.processor.decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def generate(
        self,
        image: Image.Image,
        commands: Sequence[Command],
        style_examples: Sequence[str],
        *,
        max_attempts: int = 3,
    ) -> list[str]:
        resolved = resolve_commands(commands)
        instruction = build_batch_instruction(resolved, style_examples)
        last_error: ValueError | None = None
        for _ in range(max_attempts):
            retry = (
                ""
                if last_error is None
                else f"\nThe previous response failed validation: {last_error}. Return the corrected JSON object only."
            )
            response = self._generate_once(image, instruction + retry)
            try:
                return parse_prompt_response(response, len(resolved))
            except ValueError as error:
                last_error = error

        prompts = []
        for turn_index in range(len(resolved)):
            turn_instruction = build_turn_instruction(
                resolved, turn_index, prompts, style_examples
            )
            turn_error: ValueError | None = None
            for _ in range(max_attempts):
                retry = (
                    ""
                    if turn_error is None
                    else f"\nThe previous response failed validation: {turn_error}. Return corrected JSON only."
                )
                response = self._generate_once(image, turn_instruction + retry)
                try:
                    prompts.append(parse_prompt_response(response, 1)[0])
                    break
                except ValueError as error:
                    turn_error = error
            else:
                raise RuntimeError(
                    f"prompt turn {turn_index + 1} failed after {max_attempts} attempts: {turn_error}"
                )
        return prompts