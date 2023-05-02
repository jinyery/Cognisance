#/bin/bash

python main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/baseline --require_eval --train_type baseline --phase train

python main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/center_dual --require_eval --train_type center_dual --phase train

python main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/center_dual_cos --require_eval --train_type center_dual_cos --phase train

python main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual --require_eval --train_type multi_center_dual --phase train

python main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_cos --require_eval --train_type multi_center_dual_cos --phase train

python main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_false --require_eval --train_type multi_center_dual_false --phase train

python main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_false_cos --require_eval --train_type multi_center_dual_false_cos --phase train
