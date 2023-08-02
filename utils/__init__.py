from RoPE import *
from beam_search import *

def top_p(projected_vec: torch.Tensor, top_p = 0.1):
    current_n_prob = projected_vec.squeeze(0)
    _, current_n_indexes = current_n_prob[-1].sort(descending=True)
    current_n_indexes = current_n_indexes[:int(current_n_indexes.shape[-1] * top_p)]
    next_sambol = random.choice(current_n_indexes.tolist())
    return next_sambol
