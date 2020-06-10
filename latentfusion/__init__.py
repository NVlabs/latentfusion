import logging
import os
import pathlib
import sys
import warnings

import structlog
from tqdm.auto import tqdm

package_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
resource_dir = package_dir.parent / 'resources'


def is_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def stringify_paths(logger, name, event_dict):
    for key, value in event_dict.items():
        if isinstance(value, pathlib.PurePath):
            event_dict[key] = str(value)
        if (isinstance(value, list)
                and len(value) > 0
                and isinstance(value[0], pathlib.PurePath)):
            event_dict[key] = [str(v) for v in value]

    return event_dict


class TqdmStream(object):
    @classmethod
    def write(cls, msg):
        tqdm.write(msg, end='')

    @classmethod
    def flush(cls):
        sys.stdout.flush()


if is_notebook():
    logging.basicConfig(format="%(message)s", level=logging.INFO)
else:
    logging.basicConfig(format="%(message)s", stream=TqdmStream, level=logging.INFO)
logging.getLogger("nmslib").setLevel(logging.WARNING)

structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M.%S"),
        structlog.processors.StackInfoRenderer(),
        stringify_paths,
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()  # <===
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    module='torch.nn.functional',
)
