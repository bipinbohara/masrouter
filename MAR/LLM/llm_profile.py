hosted_llm_profile = [
                # {'Name': 'gpt-4o-mini',
                #  'Description': 'GPT-4o Mini is a smaller version of the GPT-4o language model, designed for faster inference and reduced memory usage. It retains the same capabilities as the full-size model, but with fewer parameters.\n\
                #     The model costs $0.15 per million input tokens and $0.6 per million output tokens\n\
                #     In General Q&A Benchmark MMLU, GPT-4o-mini achieves an accuracy of 77.8.\n\
                #     In Reasoning Benchmark GPQA, GPT-4o-mini achieves an accuracy of 40.2.\n\
                #     In Coding Benchmark HumanEval, GPT-4o-mini achieves an accuracy of 85.7.\n\
                #     In Math Benchmark MATH, GPT-4o-mini achieves an accuracy of 66.09.'},
                # {'Name': 'claude-3-5-haiku-20241022',
                #  'Description': 'The new Claude 3.5 Haiku combines rapid response times with improved reasoning capabilities, making it ideal for tasks that require both speed and intelligence. Claude 3.5 Haiku improves on its predecessor and matches the performance of Claude 3 Opus.\n\
                #     The model costs $1.0 per million input tokens and $5.0 per million output tokens\n\
                #     In General Q&A Benchmark MMLU, claude-3-5-haiku achieves an accuracy of 67.9.\n\
                #     In Reasoning Benchmark GPQA, claude-3-5-haiku achieves an accuracy of 41.6.\n\
                #     In Coding Benchmark HumanEval, claude-3-5-haiku achieves an accuracy of 86.3.\n\
                #     In Math Benchmark MATH, claude-3-5-haiku achieves an accuracy of 65.9.'},
                # {'Name': 'gemini-1.5-flash-latest',
                #  'Description': 'Gemini 1.5 Flash was purpose-built as our fastest, most cost-efficient model yet for high volume tasks, at scale, to address developers feedback asking for lower latency and cost.\n\
                #     The model costs $0.15 per million input tokens and $0.6 per million output tokens\n\
                #     In General Q&A Benchmark MMLU, gemini-1.5-flash achieves an accuracy of 80.0.\n\
                #     In Reasoning Benchmark GPQA, gemini-1.5-flash achieves an accuracy of 39.5.\n\
                #     In Coding Benchmark HumanEval, gemini-1.5-flash achieves an accuracy of 82.6.\n\
                #     In Math Benchmark MATH, gemini-1.5-flash achieves an accuracy of 74.4.'},
                # {'Name': 'llama-3.1-70b-instruct',
                #  'Description': 'The Meta Llama-3.1-70b-instruct multilingual large language model (LLM) is a pretrained and instruction tuned generative model in 70B (text in/text out).\n\
                #     The model costs $0.2 per million input tokens and $0.2 per million output tokens\n\
                #     In General Q&A Benchmark MMLU, Llama 3.1 achieves an accuracy of 79.1.\n\
                #     In Reasoning Benchmark GPQA, Llama 3.1 achieves an accuracy of 46.7.\n\
                #     In Coding Benchmark HumanEval, Llama 3.1 achieves an accuracy of 80.7.\n\
                #     In Math Benchmark MATH, Llama 3.1 achieves an accuracy of 60.3.'},
                # {'Name': 'deepseek-chat',
                #  'Description': 'DeepSeek-V3 is a powerful open-source Mixture-of-Experts (MoE) language model developed by Chinese AI company DeepSeek, featuring 671 billion total parameters with 37 billion activated per token, achieving performance comparable to leading closed-source models like GPT-4.\n\
                #     The model costs $0.27 per million input tokens and $1.1 per million output tokens\n\
                #     In General Q&A Benchmark MMLU, deepseek-chat achieves an accuracy of 88.5.\n\
                #     In Reasoning Benchmark GPQA, deepseek-chat achieves an accuracy of 59.1.\n\
                #     In Coding Benchmark HumanEval, deepseek-chat achieves an accuracy of 88.4.\n\
                #     In Math Benchmark MATH, deepseek-chat achieves an accuracy of 85.1'},

                 {'Name': 'openai/gpt-oss-120b',
                 'Description': 'GPT-OSS 120B is an Apache 2.0-licensed text-only Mixture-of-Experts reasoning model from OpenAI with 117B total parameters and 5.1B active parameters per token.\n\
                    It has a 131,072-token context window, configurable low, medium, and high reasoning effort, and native tool-use capabilities.\n\
                    The Hugging Face checkpoint uses MXFP4 quantization for the MoE weights and is designed to fit on a single 80 GB GPU.\n\
                    Hugging Face evaluations report SWE-bench Verified scores of 47.9, 52.6, and 62.4 at low, medium, and high reasoning effort, respectively; SWE-bench Pro is 16.2 and GPQA Diamond is 80.81.\n\
                    Scores use the evaluation settings named above and should only be compared with matching benchmark versions, prompts, reasoning effort, and tool access.\n\
                    Estimated L40S compute cost: $3.16 per hour ($75.84 per day or $2,306.80 per 730-hour month) using 4 GPUs at a reference price of $0.79 per L40S GPU-hour; storage, CPU, networking, and idle-capacity overhead are excluded.\n\
                    Model card: https://huggingface.co/openai/gpt-oss-120b'},
                {'Name': 'openai/gpt-oss-20b',
                 'Description': 'GPT-OSS 20B is an Apache 2.0-licensed text-only Mixture-of-Experts reasoning model from OpenAI with 21B total parameters and 3.6B active parameters per token.\n\
                    It has a 131,072-token context window, configurable low, medium, and high reasoning effort, and native tool-use capabilities.\n\
                    The Hugging Face checkpoint uses MXFP4 quantization for the MoE weights and can run within 16 GB of memory.\n\
                    Hugging Face evaluations report SWE-bench Verified scores of 37.4, 53.2, and 60.7 at low, medium, and high reasoning effort, respectively; GPQA Diamond is 58.59 without tools and 67.1 at medium reasoning effort with tools.\n\
                    Scores use the evaluation settings named above and should only be compared with matching benchmark versions, prompts, reasoning effort, and tool access.\n\
                    Estimated L40S compute cost: $1.58 per hour ($37.92 per day or $1,153.40 per 730-hour month) using 2 GPUs at a reference price of $0.79 per L40S GPU-hour; storage, CPU, networking, and idle-capacity overhead are excluded.\n\
                    Model card: https://huggingface.co/openai/gpt-oss-20b'},
                {'Name': 'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8',
                 'Description': 'NVIDIA Nemotron 3 Super is a 120B-parameter hybrid latent Mixture-of-Experts model with approximately 12B active parameters per token, distributed in FP8 precision.\n\
                    It is a reasoning and agentic model with a hybrid Mamba-Transformer architecture and supports a context length of up to 1,048,576 tokens.\n\
                    The FP8 model scores 83.63 on MMLU-Pro, 79.36 on GPQA without tools, 78.44 on LiveCodeBench v6 (2024-08 through 2025-05), and 72.32 on IFBench Prompt.\n\
                    Additional reported results include 94.38 on HMMT Feb25 with tools, 61.07 average on TauBench V2, and 96.85/96.33/95.66 on RULER-500 at 128K/256K/512K; scores use NVIDIA\'s stated evaluation settings.\n\
                    Estimated L40S compute cost: $3.16 per hour ($75.84 per day or $2,306.80 per 730-hour month) using 4 GPUs at a reference price of $0.79 per L40S GPU-hour; storage, CPU, networking, and idle-capacity overhead are excluded.\n\
                    Model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8'},
                {'Name': 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
                 'Description': 'NVIDIA Nemotron 3 Nano is a 30B-parameter hybrid Mixture-of-Experts model with approximately 3B active parameters per token, distributed in BF16 precision.\n\
                    It uses a hybrid Mamba-Transformer architecture, is designed for reasoning and agentic workloads, and supports a context length of up to 1,048,576 tokens.\n\
                    The model scores 78.3 on MMLU-Pro, 73.0 on GPQA without tools, 68.3 on LiveCodeBench v6, and 71.5 on IFBench Prompt.\n\
                    Additional reported results include 89.1/99.2 on AIME25 without/with tools, 38.8 on SWE-Bench with OpenHands, 49.0 average on TauBench V2, and 92.9/91.3/86.3 on RULER-100 at 256K/512K/1M; scores use NVIDIA\'s stated evaluation settings.\n\
                    Estimated L40S compute cost: $1.58 per hour ($37.92 per day or $1,153.40 per 730-hour month) using 2 GPUs at a reference price of $0.79 per L40S GPU-hour; storage, CPU, networking, and idle-capacity overhead are excluded.\n\
                    Model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'},
                {'Name': 'Qwen/Qwen3.5-122B-A10B-FP8',
                 'Description': 'Qwen3.5 122B-A10B is an FP8 Mixture-of-Experts vision-language model with 122B total parameters and approximately 10B active parameters per token.\n\
                    It natively supports text, image, and video input, combines thinking and non-thinking modes, and has a 262,144-token native context window that can be extended to 1,010,000 tokens.\n\
                    Text evaluations report 86.7 on MMLU-Pro, 86.6 on GPQA Diamond, 78.9 on LiveCodeBench v6, 76.1 on IFBench, 72.0 on SWE-bench Verified, and 79.5 on TAU2-Bench.\n\
                    Vision-language evaluations report 83.9 on MMMU, 76.9 on MMMU-Pro, 86.2 on MathVision, 85.1 on RealWorldQA, and 83.9 on VideoMME without subtitles; scores use Qwen\'s stated evaluation settings.\n\
                    Estimated L40S compute cost: $3.16 per hour ($75.84 per day or $2,306.80 per 730-hour month) using 4 GPUs at a reference price of $0.79 per L40S GPU-hour; storage, CPU, networking, and idle-capacity overhead are excluded.\n\
                    Model card: https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8'},
                {'Name': 'Qwen/Qwen3.5-35B-A3B-FP8',
                 'Description': 'Qwen3.5 35B-A3B is an FP8 Mixture-of-Experts vision-language model with 35B total parameters and approximately 3B active parameters per token.\n\
                    It natively supports text, image, and video input, combines thinking and non-thinking modes, and has a 262,144-token native context window that can be extended to 1,010,000 tokens.\n\
                    Text evaluations report 85.3 on MMLU-Pro, 84.2 on GPQA Diamond, 74.6 on LiveCodeBench v6, 70.2 on IFBench, 69.2 on SWE-bench Verified, and 81.2 on TAU2-Bench.\n\
                    Vision-language evaluations report 81.4 on MMMU, 75.1 on MMMU-Pro, 83.9 on MathVision, 84.1 on RealWorldQA, and 82.5 on VideoMME without subtitles; scores use Qwen\'s stated evaluation settings.\n\
                    Estimated L40S compute cost: $1.58 per hour ($37.92 per day or $1,153.40 per 730-hour month) using 2 GPUs at a reference price of $0.79 per L40S GPU-hour; storage, CPU, networking, and idle-capacity overhead are excluded.\n\
                    Model card: https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8'},  
                ]


def get_llm_profiles(tier=None):
    """Use configured local models in every experiment, or the original hosted list."""
    from MAR.LLM.local_config import load_local_models, local_profiles
    models = load_local_models()
    if models:
        return local_profiles(models, tier)
    if tier is not None:
        raise ValueError('--llm_tier requires local model configuration')
    return hosted_llm_profile


# Preserve the public profile list for existing callers.
llm_profile = get_llm_profiles()