"""Reproducible equal-count subject sampling for held-out MMLU evaluation."""
from collections import defaultdict
import hashlib
import json


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


class SelectedMMLUDataset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.split = dataset.split

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def record_to_input(self, record):
        return self.dataset.record_to_input(record)

    def record_to_target_answer(self, record):
        return self.dataset.record_to_target_answer(record)


def balanced_mmlu_test(dataset, samples_per_subject, seed):
    """Rank stable row identities by a seeded hash; never consume router RNG."""
    if dataset.split != 'test':
        raise ValueError('Balanced MMLU sampling is only for the test split')
    if type(samples_per_subject) is not int or samples_per_subject < 1:
        raise ValueError('samples_per_subject must be a positive integer')
    groups, seen = defaultdict(list), set()
    for index in range(len(dataset)):
        row = dataset[index]
        subject = row['subject']
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError('Every MMLU row must have a nonempty subject')
        task_id = f"test:{row['source_file']}:{int(row['source_row'])}"
        if task_id in seen:
            raise ValueError(f'Duplicate MMLU source identity: {task_id}')
        seen.add(task_id)
        rank = _digest({'seed': seed, 'subject': subject, 'task_id': task_id})
        groups[subject].append((rank, task_id, index))
    if not groups:
        raise ValueError('Cannot sample an empty MMLU test split')
    short = {subject: len(rows) for subject, rows in groups.items() if len(rows) < samples_per_subject}
    if short:
        raise ValueError(f'Not enough test questions for equal sampling: requested {samples_per_subject} per subject; available={short}')
    subjects = sorted(groups)
    selected = {subject: sorted(groups[subject])[:samples_per_subject] for subject in subjects}
    # Round-robin order keeps even an interrupted prefix balanced to within one.
    ordered = [selected[subject][round_index] for round_index in range(samples_per_subject) for subject in subjects]
    indices = [index for _, _, index in ordered]
    ids = [task_id for _, task_id, _ in ordered]
    manifest = {'method': 'equal-count per subject; seeded SHA-256 ranking without replacement; round-robin order',
                'seed': seed, 'source_dataset_size': len(dataset), 'selected_size': len(indices),
                'subject_count': len(subjects), 'samples_per_subject': samples_per_subject,
                'subject_counts': {subject: samples_per_subject for subject in subjects},
                'selected_task_ids': ids, 'selection_sha256': _digest(ids)}
    return SelectedMMLUDataset(dataset, indices), manifest
