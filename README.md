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

uv pip install --python .venv/bin/python triton==3.5.0 --index-url https://pypi.org/simple

uv pip install --python .venv/bin/python triton-ascend==3.2.1 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple