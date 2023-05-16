#/bin/bash

python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/baseline --require_eval --train_type baseline --phase train

python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/center_dual --require_eval --train_type center_dual --phase train

python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual --require_eval --train_type multi_center_dual --phase train

python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_mix --require_eval --train_type multi_center_dual_mix --phase train

python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_false --require_eval --train_type multi_center_dual_false --phase train

python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_plain --require_eval --train_type multi_center_dual_plain --phase train

python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_false_plain --require_eval --train_type multi_center_dual_false_plain --phase train