import torch.multiprocessing as mp
from actor_critic import ActorCritic
from icm import ICM
from shared_adam import SharedAdam
from utils import train
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ParallelEnv:
    def __init__(self, env_id, max_epochs, input_shape, n_actions, n_threads, icm):
        names = [str(n) for n in range(n_threads+1)]

        global_actor_critic = ActorCritic(input_shape, n_actions)
        global_actor_critic.share_memory()
        global_optim = SharedAdam(global_actor_critic.parameters())

        if not icm:
            global_icm = None
            global_icm_optim = None
        else:
            global_icm = ICM(input_shape, n_actions)
            global_icm.share_memory()
            global_icm_optim = SharedAdam(global_icm.parameters())

        self.ps = [mp.Process(target=train,
                              args=(input_shape, max_epochs, n_actions,
                                    global_actor_critic, global_icm,
                                    global_optim, global_icm_optim, env_id, name, icm))
                   for name in names]

        [p.start() for p in self.ps]
        [p.join() for p in self.ps]
