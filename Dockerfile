# Headless training/inference image (no GUI) for running this project's
# ML pipeline on a CUDA-equipped server. Uses PyTorch's official CUDA
# runtime image so the CUDA/cuDNN userspace libraries match the torch
# wheel exactly - no separate CUDA toolkit install needed.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY requirements-core.txt .
# torch is already provided by the base image with the correct CUDA build;
# installing it again from plain PyPI here would silently replace it with a
# mismatched build, so it's stripped out of this in-container install.
RUN grep -v '^torch' requirements-core.txt > requirements-docker.txt \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY initialize.py .

RUN python initialize.py

ENTRYPOINT ["python"]
CMD ["scripts/train_model.py", "--help"]
