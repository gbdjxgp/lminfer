

# 安装

## GPU

```
uv pip install lminfer --torch-backend=auto

uv pip install torch -i https://mirrors.aliyun.com/pytorch-wheels/cuxxx
uv pip install lminfer
```

## NPU

```
# install triton first
uv pip uninstall --python .venv/bin/python triton triton-ascend triton_ascend || true
rm -rf .venv/lib/python3.12/site-packages/triton \
       .venv/lib/python3.12/site-packages/triton-*.dist-info \
       .venv/lib/python3.12/site-packages/triton_ascend-*.dist-info
uv pip install triton==3.5.0
uv pip install triton-ascend==3.2.1 --index-url https://pypi.org/simple \
    --find-links https://repo.huaweicloud.com/ascend/repos/pypi/triton-ascend/


# install via pypi
uv pip install lminfer[npu] --index tsinghua --index ascend --prerelease=allow
# or git clone & install -e
uv pip install -e .[npu] --index tsinghua --index ascend --prerelease=allow
```

## Verify

```
# GPU
CUDA_VISIBLE_DEVICES=0 python example.py
CUDA_VISIBLE_DEVICES=0 python bench.py
# NPU
ASCEND_RT_VISIBLE_DEVICES=0 python example.py   # 正确性
ASCEND_RT_VISIBLE_DEVICES=0 python bench.py     # 吞吐
```
