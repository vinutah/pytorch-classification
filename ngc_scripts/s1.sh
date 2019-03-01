#!/bin/bash

BATCH_SIZE=1024
LEARNING_RATE=0.0000001
L2_REG=0.0001

# Setup environment in docker instance
CMD=""
CMD+="df -h;"
CMD+="umount /dev/shm;"
CMD+="mount -t tmpfs -o rw,nosuid,nodev,noexec,relatime,size=50G shm /dev/shm;"
CMD+="df -h;"
CMD+="git clone https://swook:appkey@bitbucket.org/nvlabs/gazeseonwook;"
CMD+="cd gazeseonwook/GazeNet;"
# CMD+="git checkout baselining;"  # Switch git branch
# CMD+="git checkout $COMMIT_HASH;"  # Checkout specified commit
CMD+="apt update && apt install -y rsync ffmpeg;"
CMD+="pip install --user --upgrade -r requirements.txt;"
CMD+="cd src;"

GPU_ID=0

for METHOD in "gangliu" "xucong"; do
for ARCHITECTURE in "alexnet" "densenet"; do

	# Construct experiment identifier
	EXPERIMENT=""
	EXPERIMENT+="$METHOD"
	EXPERIMENT+="_$ARCHITECTURE"
	EXPERIMENT+="-lr$LEARNING_RATE"
	EXPERIMENT+="-bs$BATCH_SIZE"
	EXPERIMENT+="-l2$L2_REG"

	OUTPUT_DIR_STEM="/work/outputs/baselines_$(date +%m%d)"
	OUTPUT_DIR="$OUTPUT_DIR_STEM/$EXPERIMENT"

	# Copy source files to output dir
	CMD+="mkdir -p ${OUTPUT_DIR};"
	CMD+="rsync -a --include '*/' --include '*.py' --exclude '*' ./ ${OUTPUT_DIR}/src;"

	# Construct train script call
	CMD+="
		CUDA_VISIBLE_DEVICES=${GPU_ID} \
		python3 train_baselines.py \
			--base-lr $LEARNING_RATE \
			--warmup-period-for-lr 5000000 \
			--num-training-epochs 100 \
			--batch-size $BATCH_SIZE \
			--l2-reg $L2_REG \
			--print-freq-train 20 \
			--print-freq-test 2000 \
			\
			--use-tensorboard \
			--save-path $OUTPUT_DIR \
			--num-data-loaders 16 \
			--gazecapture-file /data/GazeCapture.h5 \
			--mpiigaze-file /data/MPIIFaceGaze.h5 \
			--test-subsample 0.1 \
			\
			--eval-batch-size 2048 \
			\
			$METHOD $ARCHITECTURE
	"
	CMD+="&"

	# Choose next GPU
	GPU_ID=$((GPU_ID + 1))
done
done

# Wait for all jobs to end
CMD+="wait;"

# Fix permissions
CMD+="chown -R 120000:120000 $OUTPUT_DIR;"

# Strip unnecessary whitespaces
CMD=$(echo "$CMD" | tr -s ' ' | tr  -d '\n' | tr -d '\t')
echo $CMD

# Submit job to NGC
NGC_CMD="ngc batch run \
	--name \"Baselines-lr${LEARNING_RATE}-bs${BATCH_SIZE}-l2${L2_REG}\" \
	--preempt RUNONCE \
	--ace nv-us-west-2 \
	--instance dgx1v.16g.8.norm \
	--image nvidia/pytorch:19.01-py3 \
	\
	--result /results \
	--workspace lpr-seonwook:/work:RW \
	--datasetid 23495:/data \
	\
	--org nvidian \
	--team lpr \
	\
 	--commandline \"$CMD\" \
"
echo ""
echo $NGC_CMD
eval $NGC_CMD
