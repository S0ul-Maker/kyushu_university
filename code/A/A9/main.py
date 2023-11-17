import os
import torch.multiprocessing as mp
from parallel_env import ParallelEnv

os.environ['OMP_NUM_THREADS'] = '1'

if __name__ == '__main__':
    mp.set_start_method('spawn')
    env_id = 'CartPole-v1'
    n_threads = 2
    n_actions = 2
    max_epochs = 1000
    input_shape = 4
    icm = True
    env = ParallelEnv(env_id=env_id, max_epochs=max_epochs,
                      n_actions=n_actions, input_shape=input_shape, n_threads=n_threads, icm=icm)
