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