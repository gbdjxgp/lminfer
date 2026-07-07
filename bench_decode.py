import random
import time

from lminfer import LLM, SamplingParams


def main():
    random.seed(0)
    path = "/data2/models/Qwen3-0.6B/"
    num_seqs = 512
    context_len = 256
    steps = 100

    llm = LLM(
        path,
        enforce_eager=False,
        max_model_len=context_len + steps,
        max_num_seqs=num_seqs,
        gpu_memory_utilization=0.5,
    )
    prompt_token_ids = [
        [random.randint(0, 10000) for _ in range(context_len)]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=steps)
        for _ in range(num_seqs)
    ]

    llm.generate([[0]], SamplingParams(max_tokens=1), use_tqdm=False)
    t = time.time()
    outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t

    total_tokens = sum(len(output["token_ids"]) for output in outputs)
    print(
        {
            "batch_size": num_seqs,
            "context_len": context_len,
            "steps": steps,
            "decode_tps": round(total_tokens / t, 2),
            "sec_per_step": round(t / steps, 6),
        }
    )


if __name__ == "__main__":
    main()
