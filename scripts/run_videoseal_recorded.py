#!/usr/bin/env python3
"""Run VideoSeal embedding and detection with PyAV video I/O."""

from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch

import videoseal
from videoseal.evals.metrics import bit_accuracy


def frame_rate(stream: av.video.stream.VideoStream) -> Fraction:
    rate = stream.average_rate or stream.guessed_rate
    if rate is None:
        raise RuntimeError("Input video does not declare a frame rate")
    return Fraction(rate.numerator, rate.denominator)


def embed_chunk(model, frames: list[np.ndarray], message: torch.Tensor) -> list[np.ndarray]:
    array = np.stack(frames)
    tensor = torch.from_numpy(array).float().permute(0, 3, 1, 2) / 255.0
    with torch.inference_mode():
        output = model.embed(tensor, msgs=message, is_video=True, lowres_attenuation=True)
    marked = output["imgs_w"].clamp(0, 1)
    return list((marked * 255).byte().permute(0, 2, 3, 1).cpu().numpy())


def detect_chunk(model, frames: list[np.ndarray]) -> torch.Tensor:
    array = np.stack(frames)
    tensor = torch.from_numpy(array).float().permute(0, 3, 1, 2) / 255.0
    with torch.inference_mode():
        output = model.detect(tensor, is_video=True)
    return output["preds"][:, 1:].detach().cpu()


def encode_frames(
    output_stream: av.video.stream.VideoStream,
    output_container: av.container.OutputContainer,
    frames: list[np.ndarray],
) -> None:
    for array in frames:
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in output_stream.encode(frame):
            output_container.mux(packet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--message", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--chunk-size", type=int, default=16)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("VideoSeal capture requires a CUDA GPU")

    model = videoseal.load("videoseal")
    model.eval().to("cuda")
    message = model.get_random_msg().to("cuda")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    input_container = av.open(str(args.input))
    input_stream = input_container.streams.video[0]
    rate = frame_rate(input_stream)
    width, height = input_stream.width, input_stream.height
    if width % 2 or height % 2:
        raise RuntimeError(f"VideoSeal demonstration requires even dimensions, got {width}x{height}")

    output_container = av.open(str(args.output), mode="w")
    try:
        output_stream = output_container.add_stream("libx264", rate=rate)
    except av.codec.codec.UnknownCodecError:
        output_stream = output_container.add_stream("mpeg4", rate=rate)
    output_stream.width = width
    output_stream.height = height
    output_stream.pix_fmt = "yuv420p"
    output_stream.options = {"crf": "18", "preset": "medium"}

    chunk: list[np.ndarray] = []
    input_frames = 0
    for decoded in input_container.decode(video=0):
        chunk.append(decoded.to_ndarray(format="rgb24"))
        input_frames += 1
        if len(chunk) == args.chunk_size:
            encode_frames(output_stream, output_container, embed_chunk(model, chunk, message))
            chunk.clear()
    if chunk:
        encode_frames(output_stream, output_container, embed_chunk(model, chunk, message))
    for packet in output_stream.encode():
        output_container.mux(packet)
    output_container.close()
    input_container.close()

    predictions: list[torch.Tensor] = []
    chunk.clear()
    output_frames = 0
    marked_container = av.open(str(args.output))
    for decoded in marked_container.decode(video=0):
        chunk.append(decoded.to_ndarray(format="rgb24"))
        output_frames += 1
        if len(chunk) == args.chunk_size:
            predictions.append(detect_chunk(model, chunk))
            chunk.clear()
    if chunk:
        predictions.append(detect_chunk(model, chunk))
    marked_container.close()

    # Preserve a one-element batch dimension so VideoSeal's metric interprets
    # the remaining dimensions as KxHxW message logits.
    soft_message = torch.cat(predictions, dim=0).mean(dim=0, keepdim=True)
    accuracy = float(bit_accuracy(soft_message, message.detach().cpu()).item())
    bits = "".join(str(int(value)) for value in message[0].detach().cpu())
    args.message.write_text(bits + "\n")
    metrics = {
        "model": "videoseal_1.0",
        "seed": args.seed,
        "message_bits": len(bits),
        "bit_accuracy": accuracy,
        "input_frames": input_frames,
        "output_frames": output_frames,
        "width": width,
        "height": height,
        "fps": float(rate),
    }
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"VideoSeal frames: {input_frames} input, {output_frames} output")
    print(f"VideoSeal message bits: {len(bits)}")
    print(f"VideoSeal bit accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
