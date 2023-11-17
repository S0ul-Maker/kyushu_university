import os
import torch.multiprocessing as mp
from parallel_env import ParallelEnv

os.environ['OMP_NUM_THREADS'] = '1'

if __name__ == '__main__':
    mp.set_start_method('spawn')
    env_id = 'Acrobot-v1'
    n_threads = 1
    n_actions = 3
    max_epochs = 200
    input_shape = 6
    icm = False
    env = ParallelEnv(env_id=env_id, max_epochs=max_epochs,
                      n_actions=n_actions, input_shape=input_shape, n_threads=n_threads, icm=icm)
