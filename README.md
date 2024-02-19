# Long-Tailed Classification Based on Coarse-Grained Leading Forest and Multi-Center Loss
### This Project is based on [Generalized Long-tailed Classification (GLT) Benchmarks](https://github.com/KaihuaTang/Generalized-Long-Tailed-Benchmarks.pytorch)

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
pip install joblib==1.2.0 randaugment==1.0.2 pyyaml==6.0 matplotlib==3.7.1 tqdm==4.65.0 scikit-learn==1.2.2 numpy==1.23 pandas==2.2.0

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
Run the following command to train a TRAIN_TYPE model:
```
python main.py --cfg CONFIG_PATH.yaml --output_dir OUTPUT_PATH --require_eval --train_type TRAIN_TYPE --phase train
```

### Test Models
Run the following command to test a baseline model:
```
python main.py --cfg CONFIG_PATH.yaml  --output_dir OUTPUT_PATH --require_eval --train_type TRAIN_TYPE --phase test --load_dir YOUR_CHECKPOINT.pth
```

### Train All
Run the following command to train all models:
```
bash run_all.sh -d DATASET
```
