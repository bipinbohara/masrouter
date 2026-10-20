"""Validate and recover append-only result journals before resuming evaluation."""
from datetime import datetime, timezone
import fcntl
import json
import shutil
from uuid import uuid4

from loguru import logger

# Source/config checks deliberately omit git/log/output paths: the resume
# implementation itself changes those. Keep the original run metadata intact.
IDENTITY_FIELDS = ('seed', 'split', 'dataset_size', 'source_dataset_size', 'requested_tasks',
                   'dataset_sha256', 'prompt_sha256', 'llm_profiles', 'reasoning_profiles',
                   'local_models', 'router_initialization', 'evaluated_checkpoint_sha256',
                   'checkpoint_sha256', 'evaluation_weights_source', 'test_sampling', 'device')
INTERRUPTED = 'Evaluation interrupted during this task'


def restore_writer(writer, current):
    if not writer.records_path.is_file():
        raise FileNotFoundError(f'Resume requires existing task records: {writer.records_path}')
    lock = writer.path.with_suffix('.resume.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another resume process is already writing this experiment')
    writer._resume_lock = lock
    try:
        _restore(writer, current)
    except BaseException:
        lock.close()
        writer._resume_lock = None
        raise


def _restore(writer, current):
    summary = writer.path.with_suffix('.summary.json')
    header_path = summary if summary.is_file() else writer.path
    if not header_path.is_file():
        raise FileNotFoundError('Resume requires experiment.summary.json or experiment.json metadata')
    previous = json.loads(header_path.read_text(encoding='utf-8'))['metadata']
    mismatches = [key for key in IDENTITY_FIELDS if previous.get(key) != current.get(key)]
    for package in ('torch', 'sentence-transformers', 'transformers', 'numpy'):
        if previous.get('packages', {}).get(package) != current.get('packages', {}).get(package):
            mismatches.append('packages.' + package)
    if mismatches:
        raise ValueError('Resume settings differ from original run: ' + ', '.join(mismatches))
    expected = (current.get('test_sampling') or {}).get('selected_task_ids')
    repair_offset, retry_id = None, None
    missing_newline = False
    with writer.records_path.open('rb') as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                # Only an incomplete trailing physical line is recoverable.
                if not line.endswith(b'\n') and not stream.read(1):
                    repair_offset = offset if repair_offset is None else repair_offset
                    break
                raise ValueError(f'Invalid task journal at byte {offset}; original files were not changed')
            if retry_id is not None:
                raise ValueError('An interrupted task record must be the final record')
            index = writer.summary['total']
            if record.get('evaluation_index') != index:
                raise ValueError(f'Task journal is not a unique ordered prefix at row {index}')
            task_id = record['task_id']
            if task_id in writer.completed_task_ids:
                raise ValueError(f'Duplicate saved task: {task_id}')
            if expected is not None and (index >= len(expected) or task_id != expected[index]):
                raise ValueError(f'Saved task does not match the balanced selection at row {index}')
            if record.get('error') == INTERRUPTED:
                repair_offset, retry_id = offset, task_id
                continue
            writer._accumulate(record)
            missing_newline = not line.endswith(b'\n')
    if writer.summary['total'] > current['requested_tasks']:
        raise ValueError('Saved task count exceeds requested evaluation size')

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = writer.path.parent / '.resume_backups' / f'{stamp}_{uuid4().hex[:8]}'
    backup.mkdir(parents=True, exist_ok=False)
    for path in (writer.path, writer.records_path, writer.events_path,
                 summary, writer.path.with_suffix('.xlsx')):
        if path.exists():
            shutil.copy2(path, backup / path.name)
    if repair_offset is not None:
        with writer.records_path.open('r+b') as stream:
            stream.truncate(repair_offset)
    elif missing_newline:
        with writer.records_path.open('ab') as stream:
            stream.write(b'\n')

    # Task records are authoritative. Rebuild events to recover a crash between
    # the task append and event append, without duplicating any saved calls.
    temporary = writer.events_path.with_suffix('.resume.tmp')
    with temporary.open('w', encoding='utf-8') as output, writer.records_path.open(encoding='utf-8') as source:
        for line in source:
            writer._write_events(json.loads(line), output)
    temporary.replace(writer.events_path)
    session = dict(current)
    current.clear()
    current.update(previous)
    current.pop('finished_at', None)
    current.setdefault('resume_history', []).append({
        'started_at': session['started_at'], 'git_commit': session.get('git_commit'),
        'arguments': session.get('arguments'), 'completed_before_resume': writer.summary['total'],
        'remaining_at_resume': session['requested_tasks'] - writer.summary['total'],
        'backup_directory': str(backup.resolve()), 'retried_interrupted_task_id': retry_id,
        'repaired_incomplete_tail': repair_offset is not None,
        'rng_continuity': 'New process reseeds routing; original per-task RNG states were not saved',
    })
    logger.info('RESUME: saved={} remaining={} backup={} stochastic routing restarts from the seed',
                writer.summary['total'], session['requested_tasks'] - writer.summary['total'], backup)
