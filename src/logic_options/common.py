from __future__ import annotations

import os

FOCUS_FILES_DIR = "in/focusfiles"
FOCUS_FILES_DIR_UNPRUNED = "in/focusfiles-unpruned"
FOCUS_FILES_DIR_EXTERNAL = "in/focusfiles_external"
REWARD_MODE = {
    "env": 0,
    "human": 1,
    "mixed": 2,
    None: None,
}
MULTIPROCESSING_START_METHOD = "spawn" if os.name == 'nt' else "forkserver"  # 'nt' == Windows
MODELS_BASE_PATH = "out/"
