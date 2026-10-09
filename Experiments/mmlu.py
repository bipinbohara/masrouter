"""Test evaluation independent of the original router training objective."""
import time
from MAR.Experiment.trace import collect_traces, utc_now
from MAR.Experiment.results import parse_mmlu_answer


def evaluate(router, dataset, writer, tasks, llms, reasonings, prompt_file, max_tasks=None):
    if dataset.split != 'test':
        raise ValueError('MMLU evaluation must use the test split')
    count = len(dataset) if max_tasks is None else min(max_tasks, len(dataset))
    for index in range(count):
        row = dataset[index]
        query = dataset.record_to_input(row)['task']
        truth = dataset.record_to_target_answer(row)
        record = {'task_id': f"test:{row['source_file']}:{int(row['source_row'])}",
                  'split': 'test', 'evaluation_index': index, 'subject': row['subject'],
                  'source_file': row['source_file'], 'source_row': int(row['source_row']),
                  'task': query, 'question': row['question'],
                  'options': {letter: row[letter] for letter in 'ABCD'},
                  'reference_answer': truth, 'raw_answer': None, 'predicted_answer': None,
                  'correct': False, 'status': 'error', 'agents': [], 'calls': [],
                  'started_at': utc_now(), 'api_cost_usd': None}
        start = time.perf_counter()
        with collect_traces() as traces:
            try:
                results, costs, _, _, _, agent_estimate = router.forward(
                    [query], tasks, llms, reasonings, [1], prompt_file=prompt_file)
                record.update(raw_answer=results[0], api_cost_usd=float(costs[0]),
                              predicted_agent_count=float(agent_estimate[0][0]))
                predicted = parse_mmlu_answer(results[0])
                record.update(predicted_answer=predicted, correct=predicted == truth, status='success')
            except Exception as error:
                record['error'] = f'{type(error).__name__}: {error}'
            except BaseException:
                record['error'] = 'Evaluation interrupted during this task'
                raise
            finally:
                record['task_seconds'] = time.perf_counter() - start
                record['finished_at'] = utc_now()
                if traces:
                    trace = traces[-1]
                    record.update({key: value for key, value in trace.items()
                                   if key not in ('started_at', 'finished_at', 'status', 'error', 'raw_answer')})
                    if trace.get('error'):
                        record['graph_error'] = trace['error']
                record['num_agents'] = sum(not agent['is_final_node'] for agent in record['agents'])
                record['num_nodes_including_final'] = len(record['agents'])
                record['models_used'] = sorted({call['model'] for call in record['calls']})
                record['selected_models'] = sorted({agent['model'] for agent in record['agents']})
                record['total_llm_calls'] = len(record['calls'])
                record['failed_llm_calls'] = sum(call['status'] == 'error' for call in record['calls'])
                record['had_call_errors'] = record['failed_llm_calls'] > 0
                record['known_http_requests'] = sum(call['http_requests'] or 0 for call in record['calls'])
                record['calls_without_http_count'] = sum(call['http_requests'] is None for call in record['calls'])
                record['calls_without_usage'] = sum(call['usage'] is None for call in record['calls'])
                for key in ('prompt_tokens', 'completion_tokens'):
                    record['known_' + key] = sum((call['usage'] or {}).get(key) or 0 for call in record['calls'])
                record['llm_seconds'] = sum(call['seconds'] for call in record['calls'])
                writer.append(record)
        print(f"Test {index+1}/{count}: correct={record['correct']} "
              f"agents={record['num_agents']} calls={record['total_llm_calls']} "
              f"seconds={record['task_seconds']:.3f}")