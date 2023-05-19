# Long-tail Classification Based on Invariant Feature Learning from A Multi-granularity Perspective
## This Project is based on [Generalized Long-tailed Classification (GLT) Benchmarks](https://github.com/KaihuaTang/Generalized-Long-Tailed-Benchmarks.pytorch)

## Install the Requirement
```bash
###################################
###  Step by Step Installation   ##
###################################

# 1. create and activate conda environment
conda create -n glt python=3.9
conda activate glt

# 2. install pytorch and torchvision
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# 3. install other packages
pip install joblib==1.2.0 randaugment==1.0.2 pyyaml==6.0 matplotlib==3.7.1 tqdm==4.65.0 scikit-learn==1.2.2

# 4. download this project
git clone https://github.com/jinyery/glt
```

## Prepare GLT Datasets
We propose two datasets for the Generalized Long-Tailed (GLT) classification tasks: ImageNet-GLT and MSCOCO-GLT. 
- For **ImageNet-GLT** [(link)](https://github.com/KaihuaTang/Generalized-Long-Tailed-Benchmarks.pytorch/tree/main/_ImageNetGeneration), like most of the other datasets, we don't have attribute annotations, so we use feature clusters within each class to represent K ''pretext attributes''. In other words, each cluster represents a meta attribute layout for this class.
- For **MSCOCO-GLT** [(link)](https://github.com/KaihuaTang/Generalized-Long-Tailed-Benchmarks.pytorch/tree/main/_COCOGeneration), we directly adopt attribute annotations from [MSCOCO-Attribute](https://github.com/genp/cocottributes) to construct our dataset.

Please follow the above links to prepare the datasets.

## Conduct Training and Testing

### Train Models
Run the following command to train a baseline model on Train-GLT of MSCOCO-GLT:
```
python main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/baseline --require_eval --train_type baseline --phase train
```

Run the following command to train a center_dual model on Train-GLT of MSCOCO-GLT:
```
python main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/center_dual --require_eval --train_type center_dual --phase train
```

Run the following command to train a multi_center_dual model on Train-GLT of MSCOCO-GLT:
```
python main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual --require_eval --train_type multi_center_dual --phase train
```

Run the following command to train a multi_center_dual_mix model on Train-GLT of MSCOCO-GLT:
```
python main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_mix --require_eval --train_type multi_center_dual_mix --phase train
```

Run the following command to train a multi_center_dual_false model on Train-GLT of MSCOCO-GLT:
```
python main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_false --require_eval --train_type multi_center_dual_false --phase train
```

Run the following command to train a multi_center_dual_plain model on Train-GLT of MSCOCO-GLT:
```
python main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_plain --require_eval --train_type multi_center_dual_plain --phase train
```

Run the following command to train a multi_center_dual_plain_mix model on Train-GLT of MSCOCO-GLT:
```
python main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_plain_mix --require_eval --train_type multi_center_dual_plain_mix --phase train
```

Run the following command to train a multi_center_dual_plain_false model on Train-GLT of MSCOCO-GLT:
```
python main.py --cfg config/COCO_LT.yaml --output_dir checkpoints/coco_glt/train/multi_center_dual_plain_false --require_eval --train_type multi_center_dual_plain_false --phase train
```


### Test Models
Run the following command to test a baseline model on Train-GLT of MSCOCO-GLT:
```
python main.py --cfg config/COCO_LT.yaml  --output_dir checkpoints/coco_glt/test/baseline --require_eval --train_type baseline --phase test --load_dir checkpoints/coco_glt/train/baseline/YOUR_CHECKPOINT.pth
```

Run the following command to test a center_dual model on Train-GLT of MSCOCO-GLT:
```
python main.py --cfg config/COCO_LT.yaml  --output_dir checkpoints/coco_glt/test/center_dual --require_eval --train_type center_dual --phase test --load_dir checkpoints/coco_glt/train/center_dual/YOUR_CHECKPOINT.pth
```

Run the following command to test a multi_center_dual model on Train-GLT of MSCOCO-GLT:
```
python main.py --cfg config/COCO_LT.yaml  --output_dir checkpoints/coco_glt/test/multi_center_dual --require_eval --train_type multi_center_dual --phase test --load_dir checkpoints/coco_glt/train/multi_center_dual/YOUR_CHECKPOINT.pth
```

Run the following command to test a multi_center_dual_mix model on Train-GLT of MSCOCO-GLT:
```
python main.py --cfg config/COCO_LT.yaml  --output_dir checkpoints/coco_glt/test/multi_center_dual_mix --require_eval --train_type multi_center_dual_mix --phase test --load_dir checkpoints/coco_glt/train/multi_center_dual_mix/YOUR_CHECKPOINT.pth
```

Run the following command to test a multi_center_dual_false model on Train-GLT of MSCOCO-GLT:
```
python main.py --cfg config/COCO_LT.yaml  --output_dir checkpoints/coco_glt/test/multi_center_dual_false --require_eval --train_type multi_center_dual_false --phase test --load_dir checkpoints/coco_glt/train/multi_center_dual_false/YOUR_CHECKPOINT.pth
```

Run the following command to test a multi_center_dual_plain model on Train-GLT of MSCOCO-GLT:
```
python main.py --cfg config/COCO_LT.yaml  --output_dir checkpoints/coco_glt/test/multi_center_dual_plain --require_eval --train_type multi_center_dual_plain --phase test --load_dir checkpoints/coco_glt/train/multi_center_dual_plain/YOUR_CHECKPOINT.pth
```

Run the following command to test a multi_center_dual_plain_mix model on Train-GLT of MSCOCO-GLT:
```
python main.py --cfg config/COCO_LT.yaml  --output_dir checkpoints/coco_glt/test/multi_center_dual_plain_mix --require_eval --train_type multi_center_dual_plain_mix --phase test --load_dir checkpoints/coco_glt/train/multi_center_dual_plain_mix/YOUR_CHECKPOINT.pth
```

Run the following command to test a multi_center_dual_plain_false model on Train-GLT of MSCOCO-GLT:
```
python main.py --cfg config/COCO_LT.yaml  --output_dir checkpoints/coco_glt/test/multi_center_dual_plain_false --require_eval --train_type multi_center_dual_plain_false --phase test --load_dir checkpoints/coco_glt/train/multi_center_dual_plain_false/YOUR_CHECKPOINT.pth
```
