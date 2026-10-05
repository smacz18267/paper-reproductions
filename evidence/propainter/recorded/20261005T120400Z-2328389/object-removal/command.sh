cd /home/smayan/research/ProPainter
CUDA_VISIBLE_DEVICES=1 /home/smayan/research/ProPainter/.venv/bin/python -u inference_propainter.py --video inputs/object_removal/bmx-trees --mask inputs/object_removal/bmx-trees_mask --height 240 --width 432 --fp16 --output /home/smayan/research/paper-reproductions/evidence/propainter/recorded/20261005T120400Z-2328389/object-removal/videos 
