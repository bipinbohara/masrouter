"""Incremental research artifacts with a lossless JSON source of truth."""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict


def parse_mmlu_answer(text):
    text = str(text).strip()
    boxed = re.findall(r'\\boxed\{\s*([ABCD])\s*\}', text, re.I)
    if boxed:
        return boxed[-1].upper()
    explicit = re.findall(r'(?:final\s+answer|answer)\s*(?:is|:)\s*(?:option\s*)?\(?([ABCD])\)?\b', text, re.I)
    if explicit:
        return explicit[-1].upper()
    if re.fullmatch(r'[ABCD][.)]?', text, re.I):
        return text[0].upper()
    return None


class ResultWriter:
    def __init__(self, result_file, metadata):
        self.path = Path(result_file)
        if self.path.suffix.lower() != '.json':
            raise ValueError('result_file must end in .json; Excel is exported alongside it')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records_path = self.path.with_suffix('.tasks.jsonl')
        if self.path.exists() or self.records_path.exists():
            raise FileExistsError(f'Results already exist: {self.path}; choose a new output path')
        self.metadata = metadata
        self.events_path = self.path.with_suffix('.events.jsonl')
        if self.events_path.exists():
            raise FileExistsError(f'Event log already exists: {self.events_path}')
        self.records_path.touch(exist_ok=False)
        self.events_path.touch(exist_ok=False)
        self.summary = {'total': 0, 'correct': 0, 'failed': 0, 'invalid_answers': 0,
                        'tasks_with_call_errors': 0, 'failed_llm_calls': 0, 'api_cost_usd': 0.,
                        'total_llm_calls': 0, 'graph_seconds': 0., 'task_seconds': 0.,
                        'known_http_requests': 0, 'calls_without_http_count': 0,
                        'calls_without_usage': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
        self.subjects = defaultdict(lambda: {'total': 0, 'correct': 0})
        self.models = Counter()
        self._save_sidecars('running')

    def append(self, record):
        # Close after each task so completed records survive process interruption.
        with self.records_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
        with self.events_path.open('a', encoding='utf-8') as stream:
            entries = [dict(call, event='llm_call') for call in record['calls']] + record.get('events', [])
            for event in sorted(entries, key=lambda e: e.get('started_at', e.get('timestamp', ''))):
                stream.write(json.dumps(dict(event, task_id=record['task_id']), ensure_ascii=False) + '\n')
        self.summary['node_execution_errors'] = self.summary.get('node_execution_errors', 0) + record.get('node_execution_errors', 0)
        self.summary['tasks_with_node_errors'] = self.summary.get('tasks_with_node_errors', 0) + int(record.get('node_execution_errors', 0) > 0)
        self.summary['total'] += 1
        self.summary['correct'] += int(record['correct'])
        self.summary['failed'] += int(record['status'] == 'error')
        self.summary['tasks_with_call_errors'] += int(record['had_call_errors'])
        self.summary['failed_llm_calls'] += record['failed_llm_calls']
        self.summary['api_cost_usd'] += record['api_cost_usd'] or 0.
        self.summary['invalid_answers'] += int(record['predicted_answer'] is None)
        self.summary['graph_seconds'] += record.get('graph_seconds', 0.)
        self.summary['task_seconds'] += record['task_seconds']
        subject = self.subjects[record['subject']]
        subject['total'] += 1
        subject['correct'] += int(record['correct'])
        for call in record['calls']:
            self.summary['total_llm_calls'] += 1
            self.models[call['model']] += 1
            if call['http_requests'] is None:
                self.summary['calls_without_http_count'] += 1
            else:
                self.summary['known_http_requests'] += call['http_requests']
            usage = call['usage']
            if usage is None:
                self.summary['calls_without_usage'] += 1
            else:
                for key in ('prompt_tokens', 'completion_tokens'):
                    self.summary[key] += usage.get(key) or 0
        self._save_sidecars('running')

    def _save_sidecars(self, status):
        total = self.summary['total']
        summary = dict(self.summary, accuracy=self.summary['correct'] / total if total else None,
                       macro_subject_accuracy=(sum(v['correct'] / v['total'] for v in self.subjects.values()) /
                                               len(self.subjects)) if self.subjects else None,
                       mean_graph_seconds=self.summary['graph_seconds'] / total if total else None,
                       mean_task_seconds=self.summary['task_seconds'] / total if total else None,
                       mean_llm_calls=self.summary['total_llm_calls'] / total if total else None,
                       model_call_counts=dict(self.models),
                       subjects={name: dict(values, accuracy=values['correct'] / values['total'])
                                 for name, values in self.subjects.items()})
        self.report = {'schema_version': 1, 'status': status, 'metadata': self.metadata, 'summary': summary}
        target = self.path.with_suffix('.summary.json')
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(target)

    def finish(self, status='completed', excel=False):
        self._save_sidecars(status)
        temporary = self.path.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            stream.write(json.dumps(self.report, ensure_ascii=False)[:-1] + ', "tasks": [\n')
            with self.records_path.open(encoding='utf-8') as source:
                for index, line in enumerate(source):
                    if index:
                        stream.write(',\n')
                    stream.write(line.strip())
            stream.write('\n]}\n')
        temporary.replace(self.path)
        if excel:
            self.export_excel()

    def export_excel(self):
        import pandas as pd
        def cell(value):
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            # JSON retains complete content; Excel has a 32,767-character cell limit.
            if isinstance(value, str):
                value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', value)
                if value.startswith(('=', '+', '-', '@')):
                    value = "'" + value
                if len(value) > 32767:
                    value = value[:32700] + ' [TRUNCATED: see JSON]'
            return value
        tasks, agents, calls, events, rounds = [], [], [], [], []
        with self.records_path.open(encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line)
                tasks.append({key: cell(value) for key, value in record.items() if key not in ('agents', 'calls', 'events', 'rounds')})
                for key, rows in (('agents', agents), ('calls', calls), ('events', events), ('rounds', rounds)):
                    for row in record.get(key, []):
                        rows.append(dict(task_id=record['task_id'], **{k: cell(v) for k, v in row.items()}))
        with pd.ExcelWriter(self.path.with_suffix('.xlsx'), engine='openpyxl') as workbook:
            pd.DataFrame(tasks).to_excel(workbook, sheet_name='tasks', index=False)
            pd.DataFrame(agents).to_excel(workbook, sheet_name='agents', index=False)
            # Split large call tables to stay below Excel's sheet row limit.
            for start in range(0, max(1, len(calls)), 1000000):
                pd.DataFrame(calls[start:start+1000000]).to_excel(
                    workbook, sheet_name=f'calls_{start//1000000+1}', index=False)
            for name, entries in (('events', events), ('rounds', rounds)):
                for start in range(0, max(1, len(entries)), 1000000):
                    pd.DataFrame(entries[start:start+1000000]).to_excel(
                        workbook, sheet_name=f'{name}_{start//1000000+1}', index=False)
            pd.DataFrame([{'key': key, 'value': cell(value)} for key, value in self.report.items()
                          if key != 'tasks']).to_excel(workbook, sheet_name='run_summary', index=False)
