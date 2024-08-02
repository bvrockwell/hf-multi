# FROM nvcr.io/nvidia/pytorch:24.05-py3
FROM us-east5-docker.pkg.dev/google.com/vertex-training-dlexamples/nemo-sd-training-repository/sd-accelerate_train:latest

WORKDIR /

# export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
ENV MODEL_NAME="/gcs/dlexamples-shared-data/sd3-dreambooth/models--stabilityai--stable-diffusion-3-medium-diffusers"
ENV INSTANCE_DIR="/gcs/dlexamples-shared-data/sd3-dreambooth/dog"
ENV OUTPUT_DIR="/tmp/sd3-output"

CMD ["/bin/bash"]

