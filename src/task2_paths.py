"""Account-specific Task 2 output paths, independent of Spark."""
import os
from pathlib import Path
import pwd


def scratch_root():
    """Use the actual process account, not an inherited USER environment value."""
    return (Path('/scratch') / pwd.getpwuid(os.getuid()).pw_name).resolve()


def output_path(value):
    """Allow only descendants of the executing account's scratch directory."""
    root = scratch_root()
    path = Path(value).resolve()
    if root not in path.parents:
        raise ValueError('Output must be a child directory of %s/' % root)
    return str(path)


def default_output():
    return str(scratch_root() / 'task2' / 'prepared')
