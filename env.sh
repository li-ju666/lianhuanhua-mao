# Source this on a GPU node before running the model stages:   source env.sh
# Keeps every cache next to the repository (home quotas on clusters are small) and
# works around two things we hit on GH200 (aarch64) nodes without nvcc.
export PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UV_CACHE_DIR=${UV_CACHE_DIR:-$PROJ/.cache/uv}
export UV_PROJECT_ENVIRONMENT=$PROJ/.venv
export HF_HOME=$PROJ/.cache/hf
export VLLM_CACHE_ROOT=$PROJ/.cache/vllm
export FLASHINFER_WORKSPACE_BASE=$PROJ/.cache/flashinfer
export VLLM_USE_FLASHINFER_SAMPLER=0      # FlashInfer JIT-compiles its sampler with nvcc; our nodes have none
export TOKENIZERS_PARALLELISM=false
export PATH=$HOME/.local/bin:$PATH        # where the uv installer puts uv
