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

                {
                    "Name": "openai/gpt-oss-20b",
                    "Description": "openai/gpt-oss-20b is a compact open-weight, text-only Mixture-of-Experts reasoning model with 20.9B total parameters and 3.6B active parameters per token, designed for agentic workflows with instruction following, tool use, adjustable reasoning effort, full chain-of-thought, and Structured Outputs while being small enough to run on systems with as little as 16GB of memory.\n"
                                "The deployment resource footprint for gpt-oss-20b is 2 NVIDIA L40S GPUs (96 GB aggregate GPU memory), 256 GB system memory, and 1 Intel Xeon Gold 6548Y+ CPU with 32 cores/64 threads.\n"
                                "In General Q&A Benchmark MMLU, gpt-oss-20b achieves an accuracy of 85.3% at high reasoning effort.\n"
                                "In General Q&A Benchmark MMLU-Pro, gpt-oss-20b achieves a score of 74.8%.\n"
                                "In Multilingual Knowledge Benchmark MMMLU, gpt-oss-20b achieves an average accuracy of 75.7% at high reasoning effort.\n"
                                "In Reasoning Benchmark GPQA Diamond (no tools), gpt-oss-20b achieves an accuracy of 71.5% at high reasoning effort.\n"
                                "In Reasoning Benchmark GPQA Diamond (with tools), gpt-oss-20b achieves an accuracy of 74.2% at high reasoning effort.\n"
                                "In Reasoning Benchmark Humanity's Last Exam (HLE, no tools), gpt-oss-20b achieves an accuracy of 10.9% at high reasoning effort.\n"
                                "In Reasoning Benchmark Humanity's Last Exam (HLE, with tools), gpt-oss-20b achieves an accuracy of 17.3% at high reasoning effort.\n"
                                "In Math Benchmark AIME 2024 (no tools), gpt-oss-20b achieves an accuracy of 92.1% at high reasoning effort.\n"
                                "In Math Benchmark AIME 2024 (with tools), gpt-oss-20b achieves an accuracy of 96.0% at high reasoning effort.\n"
                                "In Math Benchmark AIME 2025 (no tools), gpt-oss-20b achieves an accuracy of 91.7% at high reasoning effort.\n"
                                "In Math Benchmark AIME 2025 (with tools), gpt-oss-20b achieves an accuracy of 98.7% at high reasoning effort.\n"
                                "In Coding Benchmark SWE-Bench Verified, gpt-oss-20b achieves an accuracy of 60.7% at high reasoning effort.\n"
                                "In Coding Benchmark Aider Polyglot, gpt-oss-20b achieves an accuracy of 34.2% at high reasoning effort.\n"
                                "In Competitive Coding Benchmark Codeforces (no tools), gpt-oss-20b achieves an Elo rating of 2230 at high reasoning effort.\n"
                                "In Competitive Coding Benchmark Codeforces (with tools), gpt-oss-20b achieves an Elo rating of 2516 at high reasoning effort.\n"
                                "In Agentic Tool-Use Benchmark Tau-Bench Retail, gpt-oss-20b achieves an accuracy of 54.8% at high reasoning effort.\n"
                                "In Agentic Tool-Use Benchmark Tau-Bench Airline, gpt-oss-20b achieves an accuracy of 38.0% at high reasoning effort.\n"
                                "In Health Benchmark HealthBench, gpt-oss-20b achieves a score of 42.5% at high reasoning effort.\n"
                                "In Health Benchmark HealthBench Hard, gpt-oss-20b achieves a score of 10.8% at high reasoning effort.\n"
                                "In Health Benchmark HealthBench Consensus, gpt-oss-20b achieves a score of 82.6% at high reasoning effort."
                    },
                    {
                        "Name": "openai/gpt-oss-120b",
                        "Description": "openai/gpt-oss-120b is an open-weight, text-only Mixture-of-Experts reasoning model with 116.8B total parameters and 5.1B active parameters per token, designed for agentic workflows with strong instruction following, tool use, adjustable reasoning effort, full chain-of-thought, and Structured Outputs.\n"
                                    "The deployment resource footprint for gpt-oss-120b is 4 NVIDIA L40S GPUs (192 GB aggregate GPU memory) and one complete compute node containing 512 GB system memory and 2 Intel Xeon Gold 6548Y+ CPUs, providing 64 physical cores/128 threads in total.\n"
                                    "In General Q&A Benchmark MMLU, gpt-oss-120b achieves an accuracy of 90.0% at high reasoning effort.\n"
                                    "In General Q&A Benchmark MMLU-Pro, gpt-oss-120b achieves a score of 80.8%.\n"
                                    "In Multilingual Knowledge Benchmark MMMLU, gpt-oss-120b achieves an average accuracy of 81.3% at high reasoning effort.\n"
                                    "In Reasoning Benchmark GPQA Diamond (no tools), gpt-oss-120b achieves an accuracy of 80.1% at high reasoning effort.\n"
                                    "In Reasoning Benchmark GPQA Diamond (with tools), gpt-oss-120b achieves an accuracy of 80.9% at high reasoning effort.\n"
                                    "In Reasoning Benchmark Humanity's Last Exam (HLE, no tools), gpt-oss-120b achieves an accuracy of 14.9% at high reasoning effort.\n"
                                    "In Reasoning Benchmark Humanity's Last Exam (HLE, with tools), gpt-oss-120b achieves an accuracy of 19.0% at high reasoning effort.\n"
                                    "In Math Benchmark AIME 2024 (no tools), gpt-oss-120b achieves an accuracy of 95.8% at high reasoning effort.\n"
                                    "In Math Benchmark AIME 2024 (with tools), gpt-oss-120b achieves an accuracy of 96.6% at high reasoning effort.\n"
                                    "In Math Benchmark AIME 2025 (no tools), gpt-oss-120b achieves an accuracy of 92.5% at high reasoning effort.\n"
                                    "In Math Benchmark AIME 2025 (with tools), gpt-oss-120b achieves an accuracy of 97.9% at high reasoning effort.\n"
                                    "In Coding Benchmark SWE-Bench Verified, gpt-oss-120b achieves an accuracy of 62.4% at high reasoning effort.\n"
                                    "In Coding Benchmark Aider Polyglot, gpt-oss-120b achieves an accuracy of 44.4% at high reasoning effort.\n"
                                    "In Competitive Coding Benchmark Codeforces (no tools), gpt-oss-120b achieves an Elo rating of 2463 at high reasoning effort.\n"
                                    "In Competitive Coding Benchmark Codeforces (with tools), gpt-oss-120b achieves an Elo rating of 2622 at high reasoning effort.\n"
                                    "In Agentic Tool-Use Benchmark Tau-Bench Retail, gpt-oss-120b achieves an accuracy of 67.8% at high reasoning effort.\n"
                                    "In Agentic Tool-Use Benchmark Tau-Bench Airline, gpt-oss-120b achieves an accuracy of 49.2% at high reasoning effort.\n"
                                    "In Health Benchmark HealthBench, gpt-oss-120b achieves a score of 57.6% at high reasoning effort.\n"
                                    "In Health Benchmark HealthBench Hard, gpt-oss-120b achieves a score of 30.0% at high reasoning effort.\n"
                                    "In Health Benchmark HealthBench Consensus, gpt-oss-120b achieves a score of 89.9% at high reasoning effort."
                    },
                    {
                        "Name": "Qwen/Qwen3.5-35B-A3B-FP8",
                        "Description": "Qwen3.5-35B-A3B-FP8 is an FP8-quantized multimodal Mixture-of-Experts model with 35B total parameters and 3B activated parameters per token, combining vision-language understanding, reasoning, coding, multilingual capabilities, tool use, and agentic functionality with a native 262K-token context window.\n"
                                    "The deployment resource footprint for Qwen3.5-35B-A3B-FP8 is 4 NVIDIA L40S GPUs (192 GB aggregate GPU memory) and one complete compute node containing 512 GB system memory and 2 Intel Xeon Gold 6548Y+ CPUs, providing 64 physical cores/128 threads in total.\n"
                                    "In General Q&A Benchmark MMLU-Pro, Qwen3.5-35B-A3B-FP8 achieves a score of 85.3.\n"
                                    "In General Q&A Benchmark MMLU-Redux, Qwen3.5-35B-A3B-FP8 achieves a score of 93.3.\n"
                                    "In General Q&A Benchmark C-Eval, Qwen3.5-35B-A3B-FP8 achieves a score of 90.2.\n"
                                    "In General Reasoning Benchmark SuperGPQA, Qwen3.5-35B-A3B-FP8 achieves a score of 63.4.\n"
                                    "In Instruction-Following Benchmark IFEval, Qwen3.5-35B-A3B-FP8 achieves a score of 91.9.\n"
                                    "In Instruction-Following Benchmark IFBench, Qwen3.5-35B-A3B-FP8 achieves a score of 70.2.\n"
                                    "In Instruction-Following Benchmark MultiChallenge, Qwen3.5-35B-A3B-FP8 achieves a score of 60.0.\n"
                                    "In Long-Context Benchmark AA-LCR, Qwen3.5-35B-A3B-FP8 achieves a score of 58.5.\n"
                                    "In Long-Context Benchmark LongBench v2, Qwen3.5-35B-A3B-FP8 achieves a score of 59.0.\n"
                                    "In Reasoning Benchmark GPQA Diamond, Qwen3.5-35B-A3B-FP8 achieves a score of 84.2.\n"
                                    "In Reasoning Benchmark Humanity's Last Exam with Chain-of-Thought (HLE w/ CoT), Qwen3.5-35B-A3B-FP8 achieves a score of 22.4.\n"
                                    "In Math Benchmark HMMT February 2025, Qwen3.5-35B-A3B-FP8 achieves a score of 89.0.\n"
                                    "In Math Benchmark HMMT November 2025, Qwen3.5-35B-A3B-FP8 achieves a score of 89.2.\n"
                                    "In Coding Benchmark SWE-bench Verified, Qwen3.5-35B-A3B-FP8 achieves a score of 69.2.\n"
                                    "In Coding Benchmark LiveCodeBench v6, Qwen3.5-35B-A3B-FP8 achieves a score of 74.6.\n"
                                    "In Agentic Coding Benchmark Terminal Bench 2, Qwen3.5-35B-A3B-FP8 achieves a score of 40.5.\n"
                                    "In Competitive Coding Benchmark CodeForces, Qwen3.5-35B-A3B-FP8 achieves an Elo rating of 2028.\n"
                                    "In Coding Benchmark OJBench, Qwen3.5-35B-A3B-FP8 achieves a score of 36.0.\n"
                                    "In Full-Stack Coding Benchmark FullStackBench English, Qwen3.5-35B-A3B-FP8 achieves a score of 58.1.\n"
                                    "In Full-Stack Coding Benchmark FullStackBench Chinese, Qwen3.5-35B-A3B-FP8 achieves a score of 55.0.\n"
                                    "In Agentic Tool-Use Benchmark BFCL-V4, Qwen3.5-35B-A3B-FP8 achieves a score of 67.3.\n"
                                    "In Agentic Benchmark TAU2-Bench, Qwen3.5-35B-A3B-FP8 achieves a score of 81.2.\n"
                                    "In Agentic Benchmark VITA-Bench, Qwen3.5-35B-A3B-FP8 achieves a score of 31.9.\n"
                                    "In Agentic Planning Benchmark DeepPlanning, Qwen3.5-35B-A3B-FP8 achieves a score of 22.8.\n"
                                    "In Search-Agent Benchmark HLE with tools, Qwen3.5-35B-A3B-FP8 achieves a score of 47.4.\n"
                                    "In Search-Agent Benchmark BrowseComp, Qwen3.5-35B-A3B-FP8 achieves a score of 61.0.\n"
                                    "In Search-Agent Benchmark BrowseComp-ZH, Qwen3.5-35B-A3B-FP8 achieves a score of 69.5.\n"
                                    "In Search-Agent Benchmark WideSearch, Qwen3.5-35B-A3B-FP8 achieves a score of 57.1.\n"
                                    "In Search-Agent Benchmark Seal-0, Qwen3.5-35B-A3B-FP8 achieves a score of 41.4.\n"
                                    "In Multilingual Benchmark MMMLU, Qwen3.5-35B-A3B-FP8 achieves a score of 85.2.\n"
                                    "In Multilingual Benchmark MMLU-ProX, Qwen3.5-35B-A3B-FP8 achieves a score of 81.0.\n"
                                    "In Multilingual Benchmark NOVA-63, Qwen3.5-35B-A3B-FP8 achieves a score of 57.1.\n"
                                    "In Multilingual Benchmark INCLUDE, Qwen3.5-35B-A3B-FP8 achieves a score of 79.7.\n"
                                    "In Multilingual Commonsense Benchmark Global PIQA, Qwen3.5-35B-A3B-FP8 achieves a score of 86.6.\n"
                                    "In Multilingual Math Benchmark PolyMATH, Qwen3.5-35B-A3B-FP8 achieves a score of 64.4.\n"
                                    "In Multilingual Translation Benchmark WMT24++, Qwen3.5-35B-A3B-FP8 achieves a score of 76.3.\n"
                                    "In Multilingual Instruction-Following Benchmark MAXIFE, Qwen3.5-35B-A3B-FP8 achieves a score of 86.6."
                    },
                    {
                        "Name": "Qwen/Qwen3.5-122B-A10B-FP8",
                        "Description": "Qwen3.5-122B-A10B-FP8 is a large FP8-quantized multimodal Mixture-of-Experts model with 122B total parameters and 10B activated parameters per token, providing advanced reasoning, coding, vision-language understanding, multilingual capabilities, tool use, and agentic functionality with a native 262K-token context window.\n"
                                    "The deployment resource footprint for Qwen3.5-122B-A10B-FP8 is 2 NVIDIA L40S GPUs (96 GB aggregate GPU memory), 256 GB system memory, and 1 Intel Xeon Gold 6548Y+ CPU with 32 cores/64 threads.\n"
                                    "In General Q&A Benchmark MMLU-Pro, Qwen3.5-122B-A10B-FP8 achieves a score of 86.7.\n"
                                    "In General Q&A Benchmark MMLU-Redux, Qwen3.5-122B-A10B-FP8 achieves a score of 94.0.\n"
                                    "In General Q&A Benchmark C-Eval, Qwen3.5-122B-A10B-FP8 achieves a score of 91.9.\n"
                                    "In General Reasoning Benchmark SuperGPQA, Qwen3.5-122B-A10B-FP8 achieves a score of 67.1.\n"
                                    "In Instruction-Following Benchmark IFEval, Qwen3.5-122B-A10B-FP8 achieves a score of 93.4.\n"
                                    "In Instruction-Following Benchmark IFBench, Qwen3.5-122B-A10B-FP8 achieves a score of 76.1.\n"
                                    "In Instruction-Following Benchmark MultiChallenge, Qwen3.5-122B-A10B-FP8 achieves a score of 61.5.\n"
                                    "In Long-Context Benchmark AA-LCR, Qwen3.5-122B-A10B-FP8 achieves a score of 66.9.\n"
                                    "In Long-Context Benchmark LongBench v2, Qwen3.5-122B-A10B-FP8 achieves a score of 60.2.\n"
                                    "In Reasoning Benchmark GPQA Diamond, Qwen3.5-122B-A10B-FP8 achieves a score of 86.6.\n"
                                    "In Reasoning Benchmark Humanity's Last Exam with Chain-of-Thought (HLE w/ CoT), Qwen3.5-122B-A10B-FP8 achieves a score of 25.3.\n"
                                    "In Math Benchmark HMMT February 2025, Qwen3.5-122B-A10B-FP8 achieves a score of 91.4.\n"
                                    "In Math Benchmark HMMT November 2025, Qwen3.5-122B-A10B-FP8 achieves a score of 90.3.\n"
                                    "In Coding Benchmark SWE-bench Verified, Qwen3.5-122B-A10B-FP8 achieves a score of 72.0.\n"
                                    "In Coding Benchmark LiveCodeBench v6, Qwen3.5-122B-A10B-FP8 achieves a score of 78.9.\n"
                                    "In Agentic Coding Benchmark Terminal Bench 2, Qwen3.5-122B-A10B-FP8 achieves a score of 49.4.\n"
                                    "In Competitive Coding Benchmark CodeForces, Qwen3.5-122B-A10B-FP8 achieves an Elo rating of 2100.\n"
                                    "In Coding Benchmark OJBench, Qwen3.5-122B-A10B-FP8 achieves a score of 39.5.\n"
                                    "In Full-Stack Coding Benchmark FullStackBench English, Qwen3.5-122B-A10B-FP8 achieves a score of 62.6.\n"
                                    "In Full-Stack Coding Benchmark FullStackBench Chinese, Qwen3.5-122B-A10B-FP8 achieves a score of 58.7.\n"
                                    "In Agentic Tool-Use Benchmark BFCL-V4, Qwen3.5-122B-A10B-FP8 achieves a score of 72.2.\n"
                                    "In Agentic Benchmark TAU2-Bench, Qwen3.5-122B-A10B-FP8 achieves a score of 79.5.\n"
                                    "In Agentic Benchmark VITA-Bench, Qwen3.5-122B-A10B-FP8 achieves a score of 33.6.\n"
                                    "In Agentic Planning Benchmark DeepPlanning, Qwen3.5-122B-A10B-FP8 achieves a score of 24.1.\n"
                                    "In Search-Agent Benchmark HLE with tools, Qwen3.5-122B-A10B-FP8 achieves a score of 47.5.\n"
                                    "In Search-Agent Benchmark BrowseComp, Qwen3.5-122B-A10B-FP8 achieves a score of 63.8.\n"
                                    "In Search-Agent Benchmark BrowseComp-ZH, Qwen3.5-122B-A10B-FP8 achieves a score of 69.9.\n"
                                    "In Search-Agent Benchmark WideSearch, Qwen3.5-122B-A10B-FP8 achieves a score of 60.5.\n"
                                    "In Search-Agent Benchmark Seal-0, Qwen3.5-122B-A10B-FP8 achieves a score of 44.1.\n"
                                    "In Multilingual Benchmark MMMLU, Qwen3.5-122B-A10B-FP8 achieves a score of 86.7.\n"
                                    "In Multilingual Benchmark MMLU-ProX, Qwen3.5-122B-A10B-FP8 achieves a score of 82.2.\n"
                                    "In Multilingual Benchmark NOVA-63, Qwen3.5-122B-A10B-FP8 achieves a score of 58.6.\n"
                                    "In Multilingual Benchmark INCLUDE, Qwen3.5-122B-A10B-FP8 achieves a score of 82.8.\n"
                                    "In Multilingual Commonsense Benchmark Global PIQA, Qwen3.5-122B-A10B-FP8 achieves a score of 88.4.\n"
                                    "In Multilingual Math Benchmark PolyMATH, Qwen3.5-122B-A10B-FP8 achieves a score of 68.9.\n"
                                    "In Multilingual Translation Benchmark WMT24++, Qwen3.5-122B-A10B-FP8 achieves a score of 78.3.\n"
                                    "In Multilingual Instruction-Following Benchmark MAXIFE, Qwen3.5-122B-A10B-FP8 achieves a score of 87.9."
                    },
                    {
                        "Name": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
                        "Description": "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 is an FP8-quantized, text-only hybrid Mamba-2/Transformer Mixture-of-Experts reasoning model with 30B total parameters and approximately 3.5B active parameters per token, designed for reasoning, coding, instruction following, tool use, RAG, multilingual applications, and agentic workflows with configurable reasoning and support for context lengths up to 1M tokens.\n"
                                    "The deployment resource footprint for NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 is 2 NVIDIA L40S GPUs (96 GB aggregate GPU memory), 256 GB system memory, and 1 Intel Xeon Gold 6548Y+ CPU with 32 physical cores/64 threads.\n"
                                    "In General Q&A Benchmark MMLU-Pro, NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 achieves a score of 78.1.\n"
                                    "In Math Benchmark AIME 2025 (no tools), NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 achieves a score of 87.7.\n"
                                    "In Math Benchmark AIME 2025 (with tools), NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 achieves a score of 98.8.\n"
                                    "In Reasoning Benchmark GPQA (no tools), NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 achieves a score of 72.5.\n"
                                    "In Reasoning Benchmark GPQA (with tools), NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 achieves a score of 73.4.\n"
                                    "In Coding Benchmark LiveCodeBench v6, NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 achieves a score of 67.6.\n"
                                    "In Coding Benchmark SciCode (subtask), NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 achieves a score of 31.9.\n"
                                    "In Reasoning Benchmark Humanity's Last Exam (HLE, no tools), NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 achieves a score of 10.3.\n"
                                    "In Reasoning Benchmark Humanity's Last Exam (HLE, with tools), NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 achieves a score of 14.3.\n"
                                    "In Agentic Benchmark TauBench V2 Airline, NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 achieves a score of 44.8.\n"
                                    "In Agentic Benchmark TauBench V2 Retail, NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 achieves a score of 55.6.\n"
                                    "In Agentic Benchmark TauBench V2 Telecom, NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 achieves a score of 40.8.\n"
                                    "In Agentic Benchmark TauBench V2 Average, NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 achieves a score of 47.0.\n"
                                    "In Agentic Tool-Use Benchmark BFCL v4, NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 achieves a score of 53.2.\n"
                                    "In Instruction-Following Benchmark IFBench, NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 achieves a score of 72.2.\n"
                                    "In Long-Context Benchmark AA-LCR, NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 achieves a score of 36.1.\n"
                                    "In Multilingual Benchmark MMLU-ProX, NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 achieves an average score of 59.6."
                    },
                    {
                        "Name": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
                        "Description": "NVIDIA-Nemotron-3-Super-120B-A12B-FP8 is a large FP8-quantized, text-only hybrid Latent Mixture-of-Experts reasoning model with 120B total parameters and 12B active parameters per token, combining Mamba-2, MoE, Attention, and Multi-Token Prediction layers for advanced reasoning, coding, tool use, RAG, long-context tasks, collaborative agents, and high-volume agentic workloads with configurable reasoning and support for context lengths up to 1M tokens.\n"
                                    "The deployment resource footprint for NVIDIA-Nemotron-3-Super-120B-A12B-FP8 is 4 NVIDIA L40S GPUs (192 GB aggregate GPU memory) and one complete compute node containing 512 GB system memory and 2 Intel Xeon Gold 6548Y+ CPUs, providing 64 physical cores/128 threads in total.\n"
                                    "In General Q&A Benchmark MMLU-Pro, NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves a score of 83.63.\n"
                                    "In Math Benchmark HMMT February 2025 (with tools), NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves a score of 94.38.\n"
                                    "In Reasoning Benchmark GPQA (no tools), NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves a score of 79.36.\n"
                                    "In Coding Benchmark LiveCodeBench v6, NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves a score of 78.44.\n"
                                    "In Coding Benchmark LiveCodeBench v5, NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves a score of 80.99.\n"
                                    "In Coding Benchmark SciCode (subtask), NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves a score of 41.38.\n"
                                    "In Reasoning Benchmark Humanity's Last Exam (HLE, no tools), NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves a score of 17.42.\n"
                                    "In Agentic Coding Benchmark Terminal Bench (hard subset), NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves a score of 26.04.\n"
                                    "In Agentic Benchmark TauBench V2 Airline, NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves a score of 56.25.\n"
                                    "In Agentic Benchmark TauBench V2 Retail, NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves a score of 63.05.\n"
                                    "In Agentic Benchmark TauBench V2 Telecom, NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves a score of 63.93.\n"
                                    "In Agentic Benchmark TauBench V2 Average, NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves a score of 61.07.\n"
                                    "In Instruction-Following Benchmark IFBench, NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves a score of 72.32.\n"
                                    "In Instruction-Following Benchmark Scale AI Multi-Challenge, NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves a score of 54.35.\n"
                                    "In Chat and Instruction-Following Benchmark Arena-Hard-V2 (Hard Prompt), NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves a score of 76.06.\n"
                                    "In Long-Context Benchmark AA-LCR, NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves a score of 57.69.\n"
                                    "In Long-Context Benchmark RULER-500 at 128K context, NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves a score of 96.85.\n"
                                    "In Long-Context Benchmark RULER-500 at 256K context, NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves a score of 96.33.\n"
                                    "In Long-Context Benchmark RULER-500 at 512K context, NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves a score of 95.66.\n"
                                    "In Multilingual Benchmark MMLU-ProX, NVIDIA-Nemotron-3-Super-120B-A12B-FP8 achieves an average score of 79.21."
                    },
                ]


def get_llm_profiles(tier=None):
    """Keep the authored descriptions and order used by MasRouter's encoder."""
    from MAR.LLM.local_config import load_local_models
    if tier not in (None, 0, 1):
        raise ValueError('Local model tier must be 0 or 1')
    models = load_local_models()
    if models:
        profiles_by_name = {profile['Name']: profile for profile in hosted_llm_profile}
        missing = [name for name in models if name not in profiles_by_name]
        if missing:
            raise ValueError(f'Add Name/Description profiles in llm_profile.py for: {", ".join(missing)}')
        profiles = [profile for profile in hosted_llm_profile
                    if profile['Name'] in models and
                    (tier is None or models[profile['Name']].tier == tier)]
        if not profiles:
            raise ValueError(f'No local models configured for tier {tier}')
        return profiles
    if tier is not None:
        raise ValueError('--llm_tier requires local model configuration')
    return hosted_llm_profile


# Preserve the public profile list for existing callers.
llm_profile = get_llm_profiles()