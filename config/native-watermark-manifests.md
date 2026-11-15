# Native full-50 input manifests

VideoShield, VideoMark, SIGMark, and VidSig are coupled to video generation.
They must be evaluated on 50 authentic native watermarked generations, not on
MOSE clips with an invented post-processing adapter.

Supply one JSON file per method to `build_full50_watermark_plan.py`:

```json
{
  "method": "videoshield",
  "samples": [
    {
      "sample_id": "videoshield-000",
      "watermarked_video": "absolute-or-manifest-relative/video.mp4",
      "source_mask": "absolute-or-manifest-relative/mask.mp4",
      "method_inputs": {
        "watermark_info": "watermark-info.pt"
      },
      "provenance": {
        "prompt": "the exact generation prompt",
        "seed": 2025,
        "source_commit": "the pinned upstream commit"
      }
    }
  ]
}
```

Each manifest must contain exactly 50 unique samples. Required `method_inputs`
are: `watermark_info` for VideoShield; `shift` for VideoMark; and
`generated_output` plus `sample_key` for SIGMark. VidSig needs no per-sample
field beyond the video, mask, prompt/seed provenance, and the shared decoder in
the runtime config. Path-valued method inputs are also resolved relative to the
manifest. The planner rejects missing files, duplicate or path-traversing sample
IDs, and missing method-specific inputs before it declares a job ready.
