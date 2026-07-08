# Sandbox image for TAR method-synthesis validation.
# The synthesizer injects a generated CLMethod into sandbox_runner_template.py and runs it
# here (network-off, isolated) to verify it trains without crashing. The template imports
# torch/torch.nn/torch.nn.functional/torch.utils.data, so the image needs CPU torch.
# Built as: docker build -f scripts/sandbox_torch_cpu.Dockerfile -t tar-sandbox:torch-cpu .
FROM python:3.11-slim
# numpy from PyPI first (torch treats it as an optional dep and won't pull it via the cpu index),
# then CPU-only torch from the pytorch cpu wheel index. Generated CLMethods may import numpy.
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# smoke-test that the runtime the sandbox template needs is importable
RUN python -c "import torch, torch.nn, torch.nn.functional, torch.utils.data, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__)"
