# Concept Bottleneck Models (Resnet) - CUB Dataset
## CBMs
This code is a Concept Bottleneck Model that uses ResNet as the backbone.

## Usage
Please refer to the official Concept Bottleneck Model (CBM) code for downloading the dataset and getting started with the basic usage of the model.(https://github.com/yewsiang/ConceptBottleneck/tree/master)

```
python3 ./experiments.py cub Concept_XtoC --seed 1 -ckpt 1 -log_dir ConceptModel__Seed1/outputs/resnet18 -e 1000 -optimizer sgd -pretrained -arch resnet18 -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -bottleneck
python3 ./experiments.py cub Concept_XtoC --seed 1 -ckpt 1 -log_dir ConceptModel__Seed1/outputs/resnet34 -e 1000 -optimizer sgd -pretrained -arch resnet34 -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -bottleneck
python3 ./experiments.py cub Concept_XtoC --seed 1 -ckpt 1 -log_dir ConceptModel__Seed1/outputs/resnet50 -e 1000 -optimizer sgd -pretrained -arch resnet50 -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -bottleneck
python3 ./experiments.py cub Concept_XtoC --seed 1 -ckpt 1 -log_dir ConceptModel__Seed1/outputs/resnet101 -e 1000 -optimizer sgd -pretrained -arch resnet101 -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -bottleneck
```
