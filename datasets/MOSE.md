# MOSE as an additional object-removal test source

Source: https://github.com/henghuiding/MOSE-api
Examples: https://mose.video/

MOSE is a video object segmentation dataset/API, not an inpainting method.
It includes complex movement, occlusion, crowding and disappearance/reappearance.
MOSEv2 adds additional difficult conditions. For ProPainter qualitative tests,
start with a few training-split clips whose per-frame annotations are provided.
The public validation annotations contain only first-frame masks and therefore
need a separate segmentation/tracking step to obtain masks throughout a clip.

MOSE frames are stored in JPEGImages and object-label PNGs in Annotations.
Select one object's label ID and create a binary PNG for every frame:

```python
from PIL import Image
import numpy as np

labels = np.array(Image.open("annotation.png"))  # Keep the original label indices.
mask = ((labels == selected_object_id) * 255).astype(np.uint8)
Image.fromarray(mask).save("binary_mask.png")
```

White means remove/inpaint; black means retain. Do not convert a palette label
image to grayscale before selecting the object's ID. Match frame/mask filenames
and verify their counts and dimensions. Invisible target objects may correctly
have an empty mask in some frames.

Use the same inputs, masks and processing settings when comparing inpainting
methods, and record those decisions. MOSE annotations identify the foreground,
not the hidden background behind it: comparison against the original video
cannot establish ground-truth background reconstruction PSNR for object removal.
For reproducing ProPainter's published scores, follow its own benchmark protocol.

Store full downloaded data outside Git or under the ignored data/ directory.
MOSE data is distributed for non-commercial research under CC BY-NC-SA 4.0;
retain attribution and check the terms when publishing selected data or derivatives.
