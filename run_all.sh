#!/bin/bash

func() {
    echo "Usage:"
    echo "run_all.sh [-n N-th] [-s SEED] [-r RANGE] [-d DATASET] [-m MODE]"
    echo "Description:"
    echo "N-th, the N-th times of the task.(default to 0)"
    echo "SEED, the random number seed.(default to 25)"
    echo "RANGE, the range of task.(all/mine/plain/notplain/single)"
    echo "MODE, it's required if the RANGE is single.(default to multi_center_dual)"
    exit -1
}

N_TH=0
SEED=25
RANGE="all"
DATASET="COCO_LT"
MODE="multi_center_dual"

while getopts 'n:s:r:d:m:h' OPT; do
    case $OPT in
        n) N_TH=$OPTARG;;
        s) SEED=$OPTARG;;
        r) RANGE="$OPTARG";;
        d) DATASET="$OPTARG";;
        m) MODE="$OPTARG";;
        h) func;;
        ?) func;;
    esac
done

typeset -l OUTPUT_DIR
OUTPUT_DIR="checkpoints/$DATASET/train"
if [ $N_TH != 0 ];then
    OUTPUT_DIR="$OUTPUT_DIR/$N_TH"
fi

echo "The random number seed is $SEED"
echo "The range of task is $RANGE"
echo "The output_dir is $OUTPUT_DIR"

case $RANGE in
    "all")
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/baseline --require_eval --train_type baseline --phase train --seed $SEED
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/center_dual --require_eval --train_type center_dual --phase train --seed $SEED
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual --require_eval --train_type multi_center_dual --phase train --seed $SEED
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_mix --require_eval --train_type multi_center_dual_mix --phase train --seed $SEED
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_false --require_eval --train_type multi_center_dual_false --phase train --seed $SEED
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_plain --require_eval --train_type multi_center_dual_plain --phase train --seed $SEED
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_plain_mix --require_eval --train_type multi_center_dual_plain_mix --phase train --seed $SEED
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_plain_false --require_eval --train_type multi_center_dual_plain_false --phase train --seed $SEED;;
    "mine")
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual --require_eval --train_type multi_center_dual --phase train --seed $SEED
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_mix --require_eval --train_type multi_center_dual_mix --phase train --seed $SEED
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_false --require_eval --train_type multi_center_dual_false --phase train --seed $SEED
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_plain --require_eval --train_type multi_center_dual_plain --phase train --seed $SEED
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_plain_mix --require_eval --train_type multi_center_dual_plain_mix --phase train --seed $SEED
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_plain_false --require_eval --train_type multi_center_dual_plain_false --phase train --seed $SEED;;
    "plain")
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_plain --require_eval --train_type multi_center_dual_plain --phase train --seed $SEED
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_plain_mix --require_eval --train_type multi_center_dual_plain_mix --phase train --seed $SEED
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_plain_false --require_eval --train_type multi_center_dual_plain_false --phase train --seed $SEED;;
    "notplain")
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual --require_eval --train_type multi_center_dual --phase train --seed $SEED
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_mix --require_eval --train_type multi_center_dual_mix --phase train --seed $SEED
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_false --require_eval --train_type multi_center_dual_false --phase train --seed $SEED;;
    "single")
        python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/single/$MODE --require_eval --train_type $MODE --phase train --seed $SEED;;
    *)
        echo "Parameter RANGE can only take values from \"all\", \"mine\", \"plain\" \"notplain\" and \"single\".";;
esac
