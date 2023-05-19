#!/bin/bash

func() {
    echo "Usage:"
    echo "run_all.sh [-s SEED] [-r RANGE]"
    echo "Description:"
    echo "SEED,the random number seed.(default to 25)"
    echo "RANGE,the range of task.(all/mine/plain/notplain)"
    exit -1
}

SEED=25
RANGE="all"

while getopts 's:r:h' OPT; do
    case $OPT in
        s) SEED=$OPTARG;;
        r) RANGE="$OPTARG";;
        h) func;;
        ?) func;;
    esac
done

echo "The random number seed is $SEED"
echo "The range of task is $RANGE"

case $RANGE in
    "all")
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/baseline --require_eval --train_type baseline --phase train --seed $SEED
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/center_dual --require_eval --train_type center_dual --phase train --seed $SEED
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual --require_eval --train_type multi_center_dual --phase train --seed $SEED
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_mix --require_eval --train_type multi_center_dual_mix --phase train --seed $SEED
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_false --require_eval --train_type multi_center_dual_false --phase train --seed $SEED
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_plain --require_eval --train_type multi_center_dual_plain --phase train --seed $SEED
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_plain_mix --require_eval --train_type multi_center_dual_plain_mix --phase train --seed $SEED
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_plain_false --require_eval --train_type multi_center_dual_plain_false --phase train --seed $SEED;;
    "mine")
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual --require_eval --train_type multi_center_dual --phase train --seed $SEED
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_mix --require_eval --train_type multi_center_dual_mix --phase train --seed $SEED
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_false --require_eval --train_type multi_center_dual_false --phase train --seed $SEED
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_plain --require_eval --train_type multi_center_dual_plain --phase train --seed $SEED
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_plain_mix --require_eval --train_type multi_center_dual_plain_mix --phase train --seed $SEED
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_plain_false --require_eval --train_type multi_center_dual_plain_false --phase train --seed $SEED;;
    "plain")
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_plain --require_eval --train_type multi_center_dual_plain --phase train --seed $SEED
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_plain_mix --require_eval --train_type multi_center_dual_plain_mix --phase train --seed $SEED
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_plain_false --require_eval --train_type multi_center_dual_plain_false --phase train --seed $SEED;;
    "notplain")
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual --require_eval --train_type multi_center_dual --phase train --seed $SEED
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_mix --require_eval --train_type multi_center_dual_mix --phase train --seed $SEED
        python -u main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_false --require_eval --train_type multi_center_dual_false --phase train --seed $SEED;;
    *)
        echo "Parameter RANGE can only take values from \"all\", \"mine\", \"plain\" and \"notplain\"";;
esac
