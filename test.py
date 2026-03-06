import torch
from torch.cuda._memory_viz import save_html

save_html("memory_small_full_training.pickle", "memory_small_full_training.html")