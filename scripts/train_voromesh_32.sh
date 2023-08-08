#!/bin/bash -l

python ./src/train_SDFEnc_ME_VCDec.py ./src/configs/train_SDFEnc_ME_VCDec_32.yaml --reg_w 16.0 --n_epochs 100 --lr 0.000256 --beta2 0.99
python ./src/train_SDFEnc_ME_VCDec.py ./src/configs/train_SDFEnc_ME_VCDec_32.yaml --reg_w 4.0 --n_epochs 175 --lr 0.000064 --beta2 0.96 --resume --resume_optimizer
python ./src/train_SDFEnc_ME_VCDec.py ./src/configs/train_SDFEnc_ME_VCDec_32.yaml --reg_w 1.0 --n_epochs 200 --lr 0.000016 --beta2 0.93 --resume --resume_optimizer
python ./src/getocc_SDFEnc_ME_VCDec.py ./src/configs/getocc_SDFEnc_ME_VCDec_32.yaml
python ./src/trainocc_SDFEnc_ME_VCDec.py ./src/configs/trainocc_SDFEnc_ME_VCDec_32.yaml --n_epochs 40 --lr 0.000256 --beta2 0.99
python ./src/trainocc_SDFEnc_ME_VCDec.py ./src/configs/trainocc_SDFEnc_ME_VCDec_32.yaml --n_epochs 20 --lr 0.000064 --beta2 0.96 --resume --resume_optimizer
python ./src/trainocc_SDFEnc_ME_VCDec.py ./src/configs/trainocc_SDFEnc_ME_VCDec_32.yaml --n_epochs 15 --lr 0.000016 --beta2 0.93 --resume --resume_optimizer
