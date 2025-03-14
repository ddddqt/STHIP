import torch
import random
import numpy as np

def set_seed(seed, rank, world_size):
    rng = random.Random(seed)
    seed_per_rank = [rng.randint(0, 2**32-1) for _ in range(world_size)]
    cur_seed = seed_per_rank[rank]
    random.seed(cur_seed)
    torch.manual_seed(cur_seed)
    torch.cuda.manual_seed(cur_seed)
    np.random.seed(cur_seed)

    seed_table = {r: s for r, s in enumerate(seed_per_rank)}  # 创建随机种子表

    return seed_table, cur_seed  # 返回随机种子表和当前进程的随机种子


def set_seed_used(seed, rank, world_size, chosen_seed):
    rng = random.Random(seed)
    seed_per_rank = [rng.randint(0, 2**32-1) for _ in range(world_size)]
    cur_seed = seed_per_rank[rank]
    random.seed(cur_seed)
    torch.manual_seed(cur_seed)
    torch.cuda.manual_seed(cur_seed)
    np.random.seed(cur_seed)

    seed_table = {r: s for r, s in enumerate(seed_per_rank)}  # 创建随机种子表

    # 如果要选择特定的种子作为当前种子，可以传入 chosen_seed
    if chosen_seed is not None:
        cur_seed = chosen_seed
        random.seed(cur_seed)
        torch.manual_seed(cur_seed)
        torch.cuda.manual_seed(cur_seed)
        np.random.seed(cur_seed)

    return seed_table,cur_seed  # 返回当前进程的随机种子