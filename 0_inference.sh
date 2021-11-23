#!/bin/bash

# ------ Inference audio tagging result with pretrained model. ------
#MODEL_TYPE="Transfer_Cnn14"
#CHECKPOINT_PATH="/home/pepeotalk/panns_transfer_to_gtzan/checkpoints/main/holdout_fold=1/Transfer_Cnn14/pretrain=True/loss_type=clip_nll/augmentation=mixup/batch_size=32/freeze_base=False/10000_iterations.pth"

# Download audio tagging checkpoint.
# wget -O $CHECKPOINT_PATH "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"

## Inference.
#python3 pytorch/inference.py audio_tagging \
#    --model_type=$MODEL_TYPE \
#    --checkpoint_path=$CHECKPOINT_PATH \
#    --audio_path="/home/pepeotalk/Downloads/lef5169_20211106_190047.wav" \
#    --cuda

# ------ Inference sound event detection result with pretrained model. ------
MODEL_TYPE="Cnn14_DecisionLevelMax_Tran                                                                                                   sfer"
CHECKPOINT_PATH="/home/pepeotalk/panns_transfer_to_gtzan/checkpoints/main/holdout_fold=1/Cnn14_DecisionLevelMax_Transfer/pretrain=True/loss_type=clip_bce/augmentation=mixup/batch_size=32/freeze_base=False/10000_iterations.pth"
# Download sound event detection checkpoint.
# wget -O $CHECKPOINT_PATH "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"

# Inference.
# Inference.
python3 pytorch/inference.py audio_tagging \
    --model_type=$MODEL_TYPE \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audio_path="/home/pepeotalk/Downloads/lef5169_20211106_190047.wav" \
    --cuda

python3 pytorch/inference.py sound_event_detection \
    --model_type=$MODEL_TYPE \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audio_path="/home/pepeotalk/Downloads/lef5169_20211106_190047.wav" \
    --cuda
    