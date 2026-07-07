## install
### GPU
uv pip install lminfer --torch-backend=auto
- or:
uv pip install torch -i https://mirrors.aliyun.com/pytorch-wheels/cuxxx
uv pip install lminfer
### NPU
uv pip install lminfer[npu] --index tsinghua --index ascend --prerelease=allow

### NPU+dev:
uv pip install .[npu] --index tsinghua --index ascend --prerelease=allow


### about NPU triton install
uv pip uninstall --python .venv/bin/python triton triton-ascend triton_ascend || true

rm -rf .venv/lib/python3.12/site-packages/triton \
        .venv/lib/python3.12/site-packages/triton-*.dist-info \
        .venv/lib/python3.12/site-packages/triton_ascend-*.dist-info

uv pip install triton==3.5.0


uv pip install triton-ascend==3.2.1 --index-url https://pypi.org/simple --find-links https://repo.huaweicloud.com/ascend/repos/pypi/triton-ascend/


### about NPU+Vllm
VLLM_TARGET_DEVICE=empty uv pip install -v -e .
uv pip install -v -e . --find-links https://repo.huaweicloud.com/ascend/repos/pypi/triton-ascend/