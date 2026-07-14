import os
import time
from random import randint, seed

ENGINE = os.getenv("ENGINE", "lminfer").lower()
if ENGINE == "lminfer":
    print("use lminfer")
    from lminfer import LLM, SamplingParams
elif ENGINE == "vllm":
    from vllm import LLM, SamplingParams
else:
    raise ValueError(f"Unsupported ENGINE={ENGINE!r}")


def main():
    seed(0)
    num_seqs = int(os.getenv("BENCH_NUM_SEQS", "8"))
    max_input_len = int(os.getenv("BENCH_MAX_INPUT_LEN", "256"))
    max_ouput_len = int(os.getenv("BENCH_MAX_OUTPUT_LEN", "64"))
    min_input_len = min(100, max_input_len)
    min_output_len = min(100, max_ouput_len)
    path = os.getenv("MODEL_PATH", "/data2/models/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(min_input_len, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(min_output_len, max_ouput_len),
        )
        for _ in range(num_seqs)
    ]
    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(
        f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"
    )


if __name__ == "__main__":
    main()
