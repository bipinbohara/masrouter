"""Strict loading of router checkpoints across encoder namespace versions."""
from collections import OrderedDict
from collections.abc import Mapping
import hashlib
from pathlib import Path

from loguru import logger

_ENCODER_PREFIXES = ('text_encoder.model.0.auto_model.', 'text_encoder.model.0.model.')


def compatible_state_dict(state, target):
    if not isinstance(state, Mapping) or not all(isinstance(k, str) for k in state):
        raise ValueError('Expected a router state_dict mapping tensor names to tensors')
    result = OrderedDict()
    renamed = []
    for key, value in state.items():
        destination = key
        if key not in target:
            for source, replacement in (_ENCODER_PREFIXES, _ENCODER_PREFIXES[::-1]):
                if key.startswith(source):
                    candidate = replacement + key[len(source):]
                    if candidate in target:
                        destination = candidate
                        renamed.append((key, destination))
                        break
        if destination in result:
            raise ValueError(f'Checkpoint key collision after encoder compatibility mapping: {destination}')
        result[destination] = value
    missing, unexpected = sorted(set(target) - set(result)), sorted(set(result) - set(target))
    if missing or unexpected:
        raise ValueError(f'Incompatible router checkpoint: missing={missing[:6]}, unexpected={unexpected[:6]}. '
                         'Only the Sentence Transformers auto_model/model namespace rename is supported.')
    for key, value in result.items():
        if not hasattr(value, 'shape') or tuple(value.shape) != tuple(target[key].shape):
            raise ValueError(f'Checkpoint tensor shape mismatch: {key}: '
                             f'checkpoint={getattr(value, "shape", None)}, expected={target[key].shape}')
    if hasattr(state, '_metadata'):
        metadata = OrderedDict()
        # Module metadata uses module paths rather than parameter names.
        for key, value in state._metadata.items():
            destination = key
            if renamed:
                source, replacement = next((pair for pair in (_ENCODER_PREFIXES, _ENCODER_PREFIXES[::-1])
                                            if renamed[0][0].startswith(pair[0])))
                source, replacement = source.rstrip('.'), replacement.rstrip('.')
                if key == source or key.startswith(source + '.'):
                    destination = replacement + key[len(source):]
            if destination in metadata:
                raise ValueError(f'Checkpoint metadata collision: {destination}')
            metadata[destination] = value
        result._metadata = metadata
    return result, renamed


def load_router_checkpoint(router, path, device):
    import torch
    path = Path(path)
    # Hash and deserialize the same open file, even if its pathname is replaced.
    with path.open('rb') as stream:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
        digest = hasher.hexdigest()
        stream.seek(0)
        state = torch.load(stream, map_location=device, weights_only=True)
    target = router.state_dict()
    state, renamed = compatible_state_dict(state, target)
    if not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError('Router checkpoint must contain tensors only')
    router.load_state_dict(state, strict=True)
    info = {'path': str(path.resolve()), 'sha256': digest,
            'tensor_count': len(state), 'renamed_encoder_tensors': len(renamed)}
    logger.info('Checkpoint LOADED: path={} sha256={} tensors={} renamed_encoder_tensors={} strict=True',
                info['path'], digest, len(state), len(renamed))
    return info
