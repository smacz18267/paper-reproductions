#!/usr/bin/env python3
"""Canonical metadata for watermark methods that still need attack notebooks.

The repository already contains result notebooks for VideoSeal, EditGuard,
OmniGuard, VideoShield, and Safe-Sora.  This module records the remaining
methods from the baseline table and, importantly, distinguishes post-hoc
watermarking from watermarking that is coupled to generation.

The commit identifiers below are the upstream revisions inspected while the
notebooks were implemented.  They make the notebook templates reproducible;
users should record the exact commit they actually execute in provenance.json.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    slug: str
    label: str
    domain: str
    watermark_stage: str
    paper_url: str
    official_repository: str
    verified_commit: str
    native_entrypoint: str
    preparation_summary: str
    checkpoint_status: str
    protocol_note: str
    existing_video_compatible: bool


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        slug="rivagan",
        label="RivaGAN",
        domain="video",
        watermark_stage="post_hoc",
        paper_url="https://arxiv.org/abs/1909.01285",
        official_repository="https://github.com/DAI-Lab/RivaGAN",
        verified_commit="efffa72a4ca46d4d5051f6970c96424c2cdab441",
        native_entrypoint=(
            "from rivagan import RivaGAN; "
            "RivaGAN.load('/path/to/model.pt').encode(INPUT_VIDEO, MESSAGE, WATERMARKED_VIDEO)"
        ),
        preparation_summary=(
            "Load a trained 32-bit RivaGAN model and call its official encode method on the source video."
        ),
        checkpoint_status=(
            "BLOCKED until a trained model.pt is supplied: the official repository does not publish a "
            "pretrained checkpoint."
        ),
        protocol_note=(
            "RivaGAN can watermark an existing MOSE video, but the attack must not run until the exact "
            "trained checkpoint is recorded."
        ),
        existing_video_compatible=True,
    ),
    MethodSpec(
        slug="watermark-anything",
        label="Watermark Anything",
        domain="image (framewise video adapter)",
        watermark_stage="post_hoc",
        paper_url="https://arxiv.org/abs/2411.07231",
        official_repository="https://github.com/facebookresearch/watermark-anything",
        verified_commit="2c08af04d037d5667c02f6ddebbda9ff04581c3e",
        native_entrypoint=(
            "Use notebooks/inference_utils.py::load_model_from_checkpoint with checkpoints/params.json "
            "and wam_mit.pth; call wam.embed on every frame with one fixed 32-bit message."
        ),
        preparation_summary=(
            "Apply the official WAM embedder independently to every video frame using one fixed message, "
            "then reassemble the frames at the source frame rate."
        ),
        checkpoint_status=(
            "READY: the official MIT-licensed wam_mit.pth checkpoint is published by Meta."
        ),
        protocol_note=(
            "The paper is image-native.  The adapter must use the same message and official preprocessing "
            "for every frame and must record that this is a framewise video evaluation."
        ),
        existing_video_compatible=True,
    ),
    MethodSpec(
        slug="vine",
        label="VINE",
        domain="image (framewise video adapter)",
        watermark_stage="post_hoc",
        paper_url="https://arxiv.org/abs/2410.18775",
        official_repository="https://github.com/Shilin-LU/VINE",
        verified_commit="d72f11a93f8de9bc45ef61202b9c8dda2a70cc77",
        native_entrypoint=(
            "python vine/src/watermark_encoding_list.py --pretrained_model_name Shilin-LU/VINE-R-Enc "
            "--input_dir <frames> --output_dir <watermarked-frames> --message 'Your Secret'"
        ),
        preparation_summary=(
            "Extract the video frames, run the official VINE-R list encoder with one fixed message, and "
            "reassemble the encoded frames at the source frame rate."
        ),
        checkpoint_status=(
            "READY: VINE-R and VINE-B encoder/decoder checkpoints are published on Hugging Face."
        ),
        protocol_note=(
            "VINE center-crops non-square inputs in its demo.  The experiment must record the spatial "
            "adapter and align the object-removal mask to the watermarked frame geometry."
        ),
        existing_video_compatible=True,
    ),
    MethodSpec(
        slug="videomark",
        label="VideoMark",
        domain="video",
        watermark_stage="in_generation",
        paper_url="https://arxiv.org/abs/2504.16359",
        official_repository="https://github.com/KYRIE-LI11/VideoMark",
        verified_commit="9f8d78b73ab9f9f055651b1b4f37d68bdb05e7be",
        native_entrypoint=(
            "python embedding_and_extraction.py --model_name i2vgen-xl --num_frames 16 --num_bit 512 "
            "--num_inference_steps 50 --output_dir <output> --keys_path <keys>"
        ),
        preparation_summary=(
            "Use the authors' committed ModelScope red-panda watermarked sample, shift value, and 512-bit "
            "PRC key; then create and record one aligned removal mask."
        ),
        checkpoint_status=(
            "READY FOR ATTACK: the watermarked sample, shift, and keys are committed. Native extraction "
            "still requires the upstream ModelScope inversion model."
        ),
        protocol_note=(
            "VideoMark cannot be retrofitted onto an existing MOSE clip.  Only a natively generated "
            "VideoMark output may be passed to ProPainter or DiffuEraser."
        ),
        existing_video_compatible=False,
    ),
    MethodSpec(
        slug="sigmark",
        label="SIGMark",
        domain="video",
        watermark_stage="in_generation",
        paper_url="https://arxiv.org/abs/2603.02882",
        official_repository="https://github.com/JeremyZhao1998/SIGMark-release",
        verified_commit="3713243f2e002cb21bbc3da896094d39b2b528c9",
        native_entrypoint=(
            "python main.py --mode=gen --model_name=HunyuanVideo-I2V-community "
            "--watermark_method=sigmark --output_path=<output>"
        ),
        preparation_summary=(
            "Generate a SIGMark video with the official HunyuanVideo or HunyuanVideo-I2V integration and "
            "create a matching object mask for the generated frames."
        ),
        checkpoint_status=(
            "REQUIRES HunyuanVideo model files, at least 48 GB CPU RAM, and approximately 50 GB GPU RAM "
            "for the authors' unquantized single-GPU path. Prompt text is included in the repository."
        ),
        protocol_note=(
            "SIGMark is an in-generation watermark.  An ordinary or post-hoc-watermarked MOSE clip is not "
            "a valid substitute."
        ),
        existing_video_compatible=False,
    ),
    MethodSpec(
        slug="vidsig",
        label="VidSig / Video Signature",
        domain="video",
        watermark_stage="in_generation",
        paper_url="https://arxiv.org/abs/2506.00652",
        official_repository="https://github.com/hardenyu21/Video-Signature",
        verified_commit="9d219c47670c30fdaaccd17b24e7783c976a8c66",
        native_entrypoint="python src/generate_ms.py",
        preparation_summary=(
            "Download the authors' fine-tuned checkpoint, generate the clean and watermarked videos with "
            "src/generate_ms.py, and build a mask aligned to the watermarked result."
        ),
        checkpoint_status=(
            "REQUIRES the authors' fine-tuned checkpoint distributed through their Google Drive link."
        ),
        protocol_note=(
            "VidSig modifies the video generator.  Its watermark cannot be added faithfully to an existing "
            "MOSE video after generation."
        ),
        existing_video_compatible=False,
    ),
    MethodSpec(
        slug="lvmark",
        label="LVMark",
        domain="video",
        watermark_stage="in_generation",
        paper_url="https://arxiv.org/abs/2412.09122",
        official_repository="https://github.com/kuai-lab/TIFS26_lvmark",
        verified_commit="da7ad67ba2d86ae9918dce2b42d3736e335561be",
        native_entrypoint="cd DynamiCrafter/scripts && . ./inference_watermark.sh",
        preparation_summary=(
            "Run the official watermarked DynamiCrafter inference with the released LVMark weights, then "
            "provide a binary mask aligned to the generated video."
        ),
        checkpoint_status=(
            "REQUIRES the released LVMark weights and DynamiCrafter base checkpoint. The checked-in "
            "inference script also references a prompt-image directory and prompt/key JSON that are not "
            "present in the repository; the official weight archive index contains weight ZIPs, not those "
            "prompt files, so they must be obtained separately from the authors."
        ),
        protocol_note=(
            "LVMark jointly modifies a latent video diffusion decoder; it is not a post-hoc encoder for "
            "existing MOSE clips."
        ),
        existing_video_compatible=False,
    ),
    MethodSpec(
        slug="genptw",
        label="GenPTW",
        domain="image (framewise video adapter)",
        watermark_stage="post_hoc",
        paper_url="https://doi.org/10.1609/aaai.v40i5.37412",
        official_repository="https://github.com/GanZhenliang/GenPTW",
        verified_commit="8c22fc04dcf9d896eaeabd034d09918588e82cb3",
        native_entrypoint="python infer.py",
        preparation_summary=(
            "Use the released COCO evaluation path: VAE-encode each aligned existing frame and decode the "
            "same latent through decode_wm with one fixed 64-bit message."
        ),
        checkpoint_status=(
            "REQUIRES msg_decoder.pth, localizer.pth, and diffusion_pytorch_model.safetensors from the "
            "authors' checkpoint bundle, plus the Stable Diffusion 2 base components."
        ),
        protocol_note=(
            "GenPTW is image-native but its released training/evaluation code embeds into existing COCO "
            "images. The video experiment is therefore an explicit framewise adapter at 512 x 512."
        ),
        existing_video_compatible=True,
    ),
)


METHOD_BY_SLUG = {method.slug: method for method in METHODS}


def get_method(slug: str) -> MethodSpec:
    """Return one registered method or raise a useful error."""

    try:
        return METHOD_BY_SLUG[slug]
    except KeyError as error:
        choices = ", ".join(sorted(METHOD_BY_SLUG))
        raise ValueError(f"Unknown watermark method {slug!r}; choose one of: {choices}") from error
