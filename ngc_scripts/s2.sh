#!/bin/bash

# These parameters actually have effects!
BATCH_SIZE=2048
LEARNING_RATE=0.00005
L2_REG=0.0001

Z_DIM_APP=16
Z_DIM_GAZE=16
Z_DIM_HEAD=16
DECODER_INPUT_C=256

USE_FACE=0
USE_SKIP=0

#NORM_Z=0
#TRIPLET="euclidean"
#TRIPLET_MARGIN=0.1

#NORM_Z=1
#TRIPLET="angular"
#TRIPLET_MARGIN=0.0
#TRIPLET_REGULARIZE_WITHIN=1

NORM_Z=1
ALL_EQUAL_EMBEDDINGS=1

# Construct experiment identifier
EXPERIMENT=""
EXPERIMENT+="RotAE"
EXPERIMENT+="_a${Z_DIM_APP}"
EXPERIMENT+="_g${Z_DIM_GAZE}"
EXPERIMENT+="_h${Z_DIM_HEAD}"
EXPERIMENT+="_dc${DECODER_INPUT_C}"
EXPERIMENT+="-lr$LEARNING_RATE"
EXPERIMENT+="-bs$BATCH_SIZE"
EXPERIMENT+="-l2$L2_REG"
if [ "$USE_FACE" = "1" ]; then EXPERIMENT+="-face"; fi
if [ "$USE_SKIP" = "1" ]; then EXPERIMENT+="-skip"; fi
if [ "$NORM_Z" = "1" ]; then EXPERIMENT+="-normz"; fi
if [ "$TRIPLET" = "euclidean" ]; then EXPERIMENT+="-Teuc$TRIPLET_MARGIN"; fi
if [ "$TRIPLET" = "angular" ]; then EXPERIMENT+="-Tang$TRIPLET_MARGIN"; fi
if [ "$TRIPLET_REGULARIZE_WITHIN" = "1" ]; then EXPERIMENT+="-Treg"; fi
if [ "$ALL_EQUAL_EMBEDDINGS" = "1" ]; then EXPERIMENT+="-Zeq"; fi

# Setup environment in docker instance
CMD=""
CMD+="git clone https://swook:personal_application_key@bitbucket.org/nvlabs/gazeseonwook;"
CMD+="cd gazeseonwook/GazeNet;"
# CMD+="git checkout densenet;"  # Switch git branch
# CMD+="git checkout $COMMIT_HASH;"  # Checkout specified commit
CMD+="apt update && apt install -y ffmpeg;"
CMD+="pip install --user --upgrade -r requirements.txt;"
CMD+="cd src;"

OUTPUT_DIR_STEM="/work/outputs/rotating_autoencoder"
OUTPUT_DIR="$OUTPUT_DIR_STEM/$EXPERIMENT"

# Copy source files to output dir
CMD+="rsync -a --include '*/' --include '*.py' --exclude '*' ./ ${OUTPUT_DIR}/src;"

# Construct train script call
CMD+="python3 train_rotating_autoencoder.py \
	--base-lr $LEARNING_RATE \
	--warmup-period-for-lr 5000000 \
	--num-training-epochs 50 \
	--batch-size $BATCH_SIZE \
	--l2-reg $L2_REG \
	--print-freq-train 20 \
	--print-freq-test 2000 \
	--use-tensorboard \
	--save-path $OUTPUT_DIR \
	--num-data-loaders 64 \
	\
	--z-dim-app $Z_DIM_APP \
	--z-dim-gaze $Z_DIM_GAZE \
	--z-dim-head $Z_DIM_HEAD \
	--decoder-input-c $DECODER_INPUT_C \
	\
	--save-freq-images 10000 \
	--save-image-samples 50 \
	\
	--gazecapture-file /data/GazeCapture.h5 \
	--mpiigaze-file /data/MPIIFaceGaze.h5 \
	--test-subsample 0.2 \
"
	# --decay 0.1 \
	# --decay-interval 20000 \
if [ "$USE_FACE" = "1" ]; then CMD+=" --use-face-input"; fi
if [ "$USE_SKIP" = "1" ]; then CMD+=" --use-skip-connections"; fi
if [ "$NORM_Z" = "1" ]; then CMD+=" --normalize-3d-codes"; fi
if [ "$TRIPLET" ]; then
	CMD+=" --triplet-loss-type $TRIPLET --triplet-loss-margin $TRIPLET_MARGIN"
fi
if [ "$TRIPLET_REGULARIZE_WITHIN" = "1" ]; then CMD+=" --triplet-regularize-d-within"; fi
if [ "$ALL_EQUAL_EMBEDDINGS" = "1" ]; then CMD+=" --all-equal-embeddings"; fi
CMD+=";"

# Run meta-learning experiments
CMD+="cd ../calibration;"
META_LEARNING_CMD="python3 meta_learning.py --output-dir initial_maml"
CMD+="CUDA_VISIBLE_DEVICES=0 $META_LEARNING_CMD $OUTPUT_DIR  1 & \
      CUDA_VISIBLE_DEVICES=1 $META_LEARNING_CMD $OUTPUT_DIR  2 &
      CUDA_VISIBLE_DEVICES=2 $META_LEARNING_CMD $OUTPUT_DIR  3 &
      CUDA_VISIBLE_DEVICES=3 $META_LEARNING_CMD $OUTPUT_DIR  4 &
      CUDA_VISIBLE_DEVICES=4 $META_LEARNING_CMD $OUTPUT_DIR  5 &
      CUDA_VISIBLE_DEVICES=5 $META_LEARNING_CMD $OUTPUT_DIR  6 &
      CUDA_VISIBLE_DEVICES=6 $META_LEARNING_CMD $OUTPUT_DIR  7 &
      CUDA_VISIBLE_DEVICES=7 $META_LEARNING_CMD $OUTPUT_DIR  8 &
      CUDA_VISIBLE_DEVICES=7 $META_LEARNING_CMD $OUTPUT_DIR  9 &
      CUDA_VISIBLE_DEVICES=6 $META_LEARNING_CMD $OUTPUT_DIR 10 &
      CUDA_VISIBLE_DEVICES=5 $META_LEARNING_CMD $OUTPUT_DIR 11 &
      CUDA_VISIBLE_DEVICES=4 $META_LEARNING_CMD $OUTPUT_DIR 12 &
      CUDA_VISIBLE_DEVICES=3 $META_LEARNING_CMD $OUTPUT_DIR 13 &
      CUDA_VISIBLE_DEVICES=2 $META_LEARNING_CMD $OUTPUT_DIR 14 &
      CUDA_VISIBLE_DEVICES=1 $META_LEARNING_CMD $OUTPUT_DIR 15 &
      CUDA_VISIBLE_DEVICES=0 $META_LEARNING_CMD $OUTPUT_DIR 16;"
CMD+="wait;"
CMD+="sleep 60;"  # Maybe some file I/O are incomplete? Just wait...

# Fix permissions
CMD+="chown -R 120000:120000 $OUTPUT_DIR;"

# Strip unnecessary whitespaces
CMD=$(echo "$CMD" | tr -s ' ' | tr  -d '\n' | tr -d '\t')
echo $CMD

# Submit job to NGC
NGC_CMD="ngc batch run \
	--name \"$EXPERIMENT\" \
	--preempt RUNONCE \
	--ace nv-us-west-2 \
	--instance dgx1v.32g.8.norm \
	--image nvidia/pytorch:18.12.1-py3 \
	\
	--result /results \
	--workspace lpr-seonwook:/work:RW \
	--datasetid 23495:/data \
	\
	--org nvidian \
	--team lpr \
"
if [ "$1" ]; then
	NGC_CMD+=" --apikey $1"
fi
NGC_CMD+=" --commandline \"$CMD\""
echo ""
echo $NGC_CMD
eval $NGC_CMD
