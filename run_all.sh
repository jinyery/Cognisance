#!/bin/bash

func() {
    echo "Usage:"
    echo "run_all.sh [-n NAME] [-s SEED] [-r RANGE] [-d DATASET] [-m MODE] [-e DENOSING]"
    echo "Description:"
    echo "NAME, the Name of the task.(default to '')"
    echo "SEED, the random number seed.(default to 25)"
    echo "RANGE, the range of task.(all/mine/plain/notplain/single)"
    echo "MODE, it's required if the RANGE is single.(default to multi_center_dual)"
    echo "DENOSING, Adding this option will remove noise samples through unsupervised learning."
    echo "Aug, Adding this option will remove noise samples through unsupervised learning."
    exit 1
}

NAME=""
SEED=25
RANGE="all"
DATASET="COCO_LT"
MODE="multi_center_dual"
DENOSING=""

while getopts 'n:s:r:d:m:eh' OPT; do
    case $OPT in
    n) NAME="$OPTARG" ;;
    s) SEED=$OPTARG ;;
    r) RANGE="$OPTARG" ;;
    d) DATASET="$OPTARG" ;;
    m) MODE="$OPTARG" ;;
    e) DENOSING="--denosing" ;;
    h) func ;;
    ?) func ;;
    esac
done

typeset -l OUTPUT_DIR
OUTPUT_DIR="checkpoints/$DATASET/train"
if [ "$NAME" != '' ]; then
    OUTPUT_DIR="$OUTPUT_DIR/$NAME"
fi

echo "The random number seed is $SEED"
echo "The range of task is $RANGE"
echo "The output_dir is $OUTPUT_DIR"

case $RANGE in
"all")
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/baseline --require_eval --train_type baseline --phase train --seed $SEED $DENOSING
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/center_dual --require_eval --train_type center_dual --phase train --seed $SEED $DENOSING
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual --require_eval --train_type multi_center_dual --phase train --seed $SEED $DENOSING
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_mix --require_eval --train_type multi_center_dual_mix --phase train --seed $SEED $DENOSING
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_false --require_eval --train_type multi_center_dual_false --phase train --seed $SEED $DENOSING
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_plain --require_eval --train_type multi_center_dual_plain --phase train --seed $SEED $DENOSING
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_plain_mix --require_eval --train_type multi_center_dual_plain_mix --phase train --seed $SEED $DENOSING
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_plain_false --require_eval --train_type multi_center_dual_plain_false --phase train --seed $SEED $DENOSING
    ;;
"mine")
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual --require_eval --train_type multi_center_dual --phase train --seed $SEED $DENOSING
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_mix --require_eval --train_type multi_center_dual_mix --phase train --seed $SEED $DENOSING
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_false --require_eval --train_type multi_center_dual_false --phase train --seed $SEED $DENOSING
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_plain --require_eval --train_type multi_center_dual_plain --phase train --seed $SEED $DENOSING
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_plain_mix --require_eval --train_type multi_center_dual_plain_mix --phase train --seed $SEED $DENOSING
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_plain_false --require_eval --train_type multi_center_dual_plain_false --phase train --seed $SEED $DENOSING
    ;;
"plain")
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_plain --require_eval --train_type multi_center_dual_plain --phase train --seed $SEED $DENOSING
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_plain_mix --require_eval --train_type multi_center_dual_plain_mix --phase train --seed $SEED $DENOSING
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_plain_false --require_eval --train_type multi_center_dual_plain_false --phase train --seed $SEED $DENOSING
    ;;
"notplain")
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual --require_eval --train_type multi_center_dual --phase train --seed $SEED $DENOSING
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_mix --require_eval --train_type multi_center_dual_mix --phase train --seed $SEED $DENOSING
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/multi_center_dual_false --require_eval --train_type multi_center_dual_false --phase train --seed $SEED $DENOSING
    ;;
"single")
    python -u main.py --cfg config/$DATASET.yaml --output_dir $OUTPUT_DIR/single/$MODE --require_eval --train_type $MODE --phase train --seed $SEED $DENOSING
    ;;
*)
    echo "Parameter RANGE can only take values from \"all\", \"mine\", \"plain\" \"notplain\" and \"single\"."
    ;;
esac
