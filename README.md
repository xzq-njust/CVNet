# CVNet: Contour Vibratioin Network for Building Extraction #

This repository contains code for the CVNet framework as described in our [CVPR 2022 paper].

## Dependencies
- Python 3 (we used v3.6.5)
- PyTorch (we used v1.2.0)
- PIL
- scipy and associated packages
- tqdm
- ChamferDistance

## Instructions
1. Download datasets from [here](https://drive.google.com/file/d/1Ug4HuH7wHH6xbKB-UHxyDlJHcmLQTroL/view?usp=sharing)
1. Unzip datasets into your desired directory `datasets`
1. Modify these directories in `package/config/config.ini`, and the directory where you intend to keep results
1. Complete train/val/test split by `split_datasets.py`, our divided txt files are placed in `package/datasets`
1. Run experiments with `runner.py` 
1. Download models from [here](https://drive.google.com/file/d/132_AnOKSc5jC1s2qnBRBjLxxzlXlMv60/view?usp=sharing)

## References
1. Darnet (https://github.com/dcheng-utoronto/darnet)
1. ChamferDistance (https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)

Please contact me at xuziqiang@njust.edu.cn for questions.