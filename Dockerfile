# FROM nvcr.io/nvidia/pytorch:24.05-py3
FROM us-east5-docker.pkg.dev/google.com/vertex-training-dlexamples/nemo-sd-training-repository/sd-accelerate_train:latest

WORKDIR /

RUN git clone https://github.com/bvrockwell/hf-multi.git

ENV PYTHONPATH ./
ENV NCCL_LIB_DIR="/usr/local/nvidia/lib64"
ENV NCCL_FASTRAK_IFNAME=eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8
ENV NCCL_FASTRAK_CTRL_DEV=eth0
ENV NCCL_SOCKET_IFNAME=eth0
ENV NCCL_CROSS_NIC=0
ENV NCCL_ALGO=Ring,Tree
ENV NCCL_PROTO=Simple
ENV NCCL_MIN_NCHANNELS=4
ENV NCCL_P2P_NET_CHUNKSIZE=524288
ENV NCCL_P2P_PCI_CHUNKSIZE=524288
ENV NCCL_P2P_NVL_CHUNKSIZE=1048576
ENV NCCL_FASTRAK_NUM_FLOWS=2
ENV NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL=0
ENV NCCL_BUFFSIZE=8388608
ENV NCCL_FASTRAK_USE_SNAP=1
ENV NCCL_FASTRAK_USE_LLCM=1
ENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV NCCL_NET_GDR_LEVEL=PIX
ENV NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING=0
ENV NCCL_TUNER_PLUGIN=libnccl-tuner.so
ENV NCCL_TUNER_CONFIG_PATH=${NCCL_LIB_DIR}/a3plus_tuner_config.textproto
ENV NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=${NCCL_LIB_DIR}/a3plus_guest_config.textproto
ENV NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000
ENV NCCL_NVLS_ENABLE=0 

ENV NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=$NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS
ENV MASTER_ADDR = $(if [[ $RANK -gt 0 ]]; then echo $MASTER_ADDR;else echo localhost;fi)
ENV MASTER_PORT=$MASTER_PORT
ENV NUM_PROCESSES = $((NODE_COUNT * $NUM_PROCESS))

ENV MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
ENV MODEL_NAME="/gcs/dlexamples-shared-data/sd3-dreambooth/models--stabilityai--stable-diffusion-3-medium-diffusers"
ENV INSTANCE_DIR="/gcs/dlexamples-shared-data/sd3-dreambooth/dog"
ENV OUTPUT_DIR="/tmp/sd3-output"
ENV ACC_CONFIG="${ACC_CONFIG:-/hf-multi/2host_config.yaml}"

# update config for # of nodes
RUN ["/bin/bash", "-c", "./opt/conda/bin/accelerate config update --config_file $ACC_CONFIG"]

ENV LAUNCHER="./opt/conda/bin/accelerate launch \
    --num_processes $NUM_PROCESSES \
    --num_machines $NODE_COUNT \
    --rdzv_backend c10d \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $RANK \
    "

ENV PYTHON_SCRIPT="diffusers/examples/dreambooth/train_dreambooth_sd3.py"

ENV SCRIPT_ARGS=" \
    --debug \
    --pretrained_model_name_or_path $MODEL_NAME  \
    --instance_data_dir $INSTANCE_DIR \
    --output_dir $OUTPUT_DIR \
    --mixed_precision fp16 \
    --instance_prompt 'a photo of sks dog' \
    --resolution 1024 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lr_scheduler constant \
    --lr_warmup_steps 0 \
    --max_train_steps 500 \
    --validation_prompt 'A photo of sks dog in a bucket' \
    --validation_epochs 25 \
    --seed 0 \
    "

# This step is necessary because accelerate launch does not handle multiline arguments properly
ENV LAUNCH_CMD="$LAUNCHER $PYTHON_FILE $ARGS" 

CMD ["/bin/bash", "-c", "$LAUNCH_CMD"]

