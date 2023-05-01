# MSCOCO-GLT Dataset Generation

## For details, please refer to: [MSCOCO-GLT Dataset Generation](https://github.com/KaihuaTang/Generalized-Long-Tailed-Benchmarks.pytorch/tree/main/_COCOGeneration)

## Commands
Run the following command to generate the MSCOCO-GLT annotations: 1) coco_intra_lt_inter_lt.json and 2) coco_intra_lt_inter_bl.json
```
python mscoco_glt_generation.py --data_path YOUR_COCO_IMAGE_FOLDER --anno_path YOUR_COCO_ANNOTATION_FOLDER --attribute_path ./cocottributes_py3.jbl
```
Run the following command to crop objects from MSCOCO images
```
python mscoco_glt_crop.py --data_path YOUR_COCO_IMAGE_FOLDER --output_path OUTOUT_FOLDER
```
