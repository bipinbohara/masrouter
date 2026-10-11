"""Experiment training budget; independent of routing and reward policy."""


def training_limit_reached(args, samples):
    return args.max_train_samples > 0 and samples >= args.max_train_samples


def limit_training_batch(batch, args, samples):
    if args.max_train_samples == 0:
        return batch
    # Works with the runners' lists and MMLU's positional DataFrame slice.
    return batch[:max(0, args.max_train_samples - samples)]

def grade_training_answer(benchmark, response, original, timeout=100):
    """Use benchmark grading for rewards; invalid outputs score zero, not a crash.

    This changes answer checking only. Routing, node retries and loss equations
    stay in their existing runners. KeyboardInterrupt/SystemExit propagate.
    """
    from MAR.Experiment.results import parse_mmlu_answer
    from MAR.Experiment.benchmarks import score
    if not isinstance(response, str) or not response.strip():
        return {'predicted_answer': None, 'correct': False, 'grading_error': 'Empty or non-string model answer'}
    try:
        if benchmark == 'mmlu':
            predicted = parse_mmlu_answer(response)
            return {'predicted_answer': predicted,
                    'correct': predicted is not None and predicted == str(original['answer']).strip().upper()}
        return score(benchmark, response, {'original': original}, timeout)
    except (ImportError, MemoryError):
        raise
    except Exception as error:
        return {'predicted_answer': None, 'correct': False,
                'grading_error': f'{type(error).__name__}: {error}'}
