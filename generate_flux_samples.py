"""Generate test images using Flux Schnell for detector evaluation.

Flux Schnell (Apache 2.0) is used because it's the only fully open-source
Flux variant.  It generates in 1-4 steps via adversarial distillation,
which makes it fast but introduces distillation-specific artifacts.

Requirements
------------
- torch >= 2.0
- diffusers >= 0.30.0
- transformers
- accelerate
- sentencepiece

GPU memory: ~12GB with float16, ~24GB with float32.
Use --cpu for CPU inference (very slow, ~10 min/image).

Usage
-----
    python generate_flux_samples.py --output data/generated/flux --num 20
    python generate_flux_samples.py --output data/generated/flux --num 5 --steps 4
    python generate_flux_samples.py --output data/generated/flux --num 5 --cpu
"""

import argparse
import json
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate test images with Flux Schnell"
    )
    parser.add_argument(
        "--output", type=str, default="data/generated/flux",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--num", type=int, default=20,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--steps", type=int, default=4,
        help="Number of inference steps (Schnell: 1-4)",
    )
    parser.add_argument(
        "--width", type=int, default=512,
        help="Image width",
    )
    parser.add_argument(
        "--height", type=int, default=512,
        help="Image height",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU inference (slow)",
    )
    return parser.parse_args()


# Diverse prompts covering Flux's strengths and known weaknesses
PROMPTS = [
    # Portraits (primary detection target)
    "professional headshot of a young woman, studio lighting, neutral background",
    "portrait of a middle-aged man with glasses, natural light, shallow depth of field",
    "close-up portrait of an elderly person, wrinkles and skin texture visible",
    "candid photo of a person laughing, outdoor setting, bokeh background",
    "formal corporate headshot, suit and tie, grey backdrop",
    # Faces with challenging attributes
    "portrait with dramatic side lighting, half face in shadow",
    "person with wet hair, water droplets on skin, high detail",
    "person wearing a hat, partial face occlusion, street photography style",
    # Scenes (for non-face artifact analysis)
    "aerial view of a city at sunset, golden hour lighting",
    "macro photograph of a flower with water droplets",
    "photorealistic interior of a modern kitchen, natural light",
    "street photography, busy intersection, motion blur on pedestrians",
    # Text rendering (known Flux weakness — good for artifact detection)
    "storefront with visible sign reading OPEN, street view",
    "close-up of a handwritten note on lined paper",
    "newspaper front page with headline text visible",
    # Complex scenes (stress test for attention mechanism)
    "group photo of five people at a dinner table, restaurant setting",
    "crowded market scene with many faces visible",
    "reflection of a person in a mirror, bathroom setting",
    "person holding a phone showing their own face on screen",
    "two people shaking hands, close-up of hands and faces",
]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import to fail fast on missing dependencies
    try:
        from diffusers import FluxPipeline
    except ImportError:
        print("ERROR: diffusers >= 0.30.0 required for Flux support.")
        print("Install with: pip install diffusers[torch] transformers accelerate sentencepiece")
        return

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Loading Flux Schnell...")

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=dtype,
    )

    if device != "cpu":
        pipe.to(device)
    else:
        # CPU: enable sequential offloading to reduce memory
        pipe.enable_sequential_cpu_offload()

    prompts = PROMPTS[:args.num] if args.num <= len(PROMPTS) else (
        PROMPTS * (args.num // len(PROMPTS) + 1)
    )[:args.num]

    generator = torch.Generator(device=device).manual_seed(args.seed)
    metadata = []

    print(f"Generating {len(prompts)} images at {args.width}x{args.height}, "
          f"{args.steps} steps...\n")

    for idx, prompt in enumerate(prompts):
        t0 = time.time()

        result = pipe(
            prompt=prompt,
            num_inference_steps=args.steps,
            width=args.width,
            height=args.height,
            generator=generator,
        )
        image = result.images[0]

        filename = f"flux_schnell_{idx:03d}.png"
        filepath = output_dir / filename
        image.save(filepath)

        elapsed = time.time() - t0
        meta = {
            "filename": filename,
            "prompt": prompt,
            "model": "FLUX.1-schnell",
            "steps": args.steps,
            "width": args.width,
            "height": args.height,
            "seed": args.seed,
            "generation_time_s": round(elapsed, 2),
        }
        metadata.append(meta)
        print(f"  [{idx + 1}/{len(prompts)}] {filename} ({elapsed:.1f}s) — {prompt[:50]}...")

    # Save generation metadata
    meta_path = output_dir / "generation_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. {len(metadata)} images saved to {output_dir}/")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
