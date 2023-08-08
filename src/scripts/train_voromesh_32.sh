#!/bin/bash -l

python train_SDFEnc_ME_VCDec.py ./config/train_SDFEnc_ME_VCDec_32.yaml --reg_w 16.0 --n_epochs 100 --lr 0.000256 --beta2 0.99
python train_SDFEnc_ME_VCDec.py ./config/train_SDFEnc_ME_VCDec_32.yaml --reg_w 4.0 --n_epochs 175 --lr 0.000064 --beta2 0.96 --resume --resume_optimizer
python train_SDFEnc_ME_VCDec.py ./config/train_SDFEnc_ME_VCDec_32.yaml --reg_w 1.0 --n_epochs 200 --lr 0.000016 --beta2 0.93 --resume --resume_optimizer
python getocc_SDFEnc_ME_VCDec.py ./config/getocc_SDFEnc_ME_VCDec_32.yaml
python trainocc_SDFEnc_ME_VCDec.py ./config/trainocc_SDFEnc_ME_VCDec_32.yaml --n_epochs 40 --lr 0.000256 --beta2 0.99
python trainocc_SDFEnc_ME_VCDec.py ./config/trainocc_SDFEnc_ME_VCDec_32.yaml --n_epochs 20 --lr 0.000064 --beta2 0.96 --resume --resume_optimizer
python trainocc_SDFEnc_ME_VCDec.py ./config/trainocc_SDFEnc_ME_VCDec_32.yaml --n_epochs 15 --lr 0.000016 --beta2 0.93 --resume --resume_optimizer
