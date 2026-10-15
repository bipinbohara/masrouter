"""A dedicated file sink for shared benchmark tests, including startup failures."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from loguru import logger

LOG_DIR = Path(__file__).resolve().parents[2] / 'logs'


@contextmanager
def experiment_file_log(benchmark):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = LOG_DIR / f'{benchmark}_test_{stamp}_{uuid4().hex[:8]}.log'
    sink = logger.add(path, level='DEBUG', encoding='utf-8')
    try:
        logger.info('Test log file: {}', path)
        yield path
    except BaseException:
        logger.exception('Test run stopped: benchmark={}', benchmark)
        raise
    finally:
        logger.remove(sink)
