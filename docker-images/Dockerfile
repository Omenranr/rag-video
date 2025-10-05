# syntax=docker/dockerfile:1.7
FROM vastai/vllm:v0.10.2-cuda-12.9-pytorch-2.8.0-py312

# Cache dirs + add ~/.local/bin to PATH
ENV HF_HOME=/opt/hf \
    HUGGINGFACE_HUB_CACHE=/opt/hf \
    VLLM_DOWNLOAD_DIR=/workspace/models \
    PATH="/root/.local/bin:${PATH}"

# Tools to download from Hugging Face (installed for python3)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir --upgrade huggingface_hub hf_transfer

# Target directory for models
RUN mkdir -p /workspace/models

# Optional: HF token for gated models (pass at build with --build-arg HF_TOKEN=hf_xxx)
ARG HF_TOKEN

# Pre-download the model (avoid heredoc to be CRLF-safe on Windows)
RUN HF_TOKEN="${HF_TOKEN}" python3 -c "import os; \
from huggingface_hub import snapshot_download; \
model_id='OpenGVLab/InternVL3-14B'; \
local_dir='/workspace/models/OpenGVLab/InternVL3-14B'; \
token=os.getenv('HF_TOKEN') or None; \
snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False, token=token, max_workers=8); \
print('Downloaded:', model_id, '->', local_dir)"

# (Optional) quick sanity check; safe to remove later
RUN python3 -c "import os; p='/workspace/models/OpenGVLab/InternVL3-14B'; \
print('Exists:', os.path.isdir(p), 'Files:', len(list(__import__('pathlib').Path(p).rglob('*'))))"
