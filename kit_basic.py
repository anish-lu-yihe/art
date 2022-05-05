from functools import partial
import numpy as np


l1_norm = partial(np.linalg.norm, ord=1, axis=-1)