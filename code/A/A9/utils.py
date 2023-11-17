import gym
import numpy as np
import torch
from actor_critic import ActorCritic
from icm import ICM
from memory import Memory
import matplotlib.pyplot as plt
torch.manual_seed(3)
np.random.seed(3)


def train(input_shape, max_epochs, n_actions, global_agent, global_icm,
          optimizer, icm_optimizer, env_id, name, icm=False):
    T_MAX = 20
    STEP_MAX = 5000

    if icm:
        local_icm = ICM(input_shape, n_actions)
        algo = 'ICM'
    else:
        intrinsic_reward = torch.zeros(1)
        algo = 'A2C'

    local_agent = ActorCritic(input_shape, n_actions)
    memory = Memory()
    env = gym.make(env_id, render_mode="rgb_array")

    episode, scores_list, avg_score = 0, [], 0.0

    while episode < max_epochs:
        obs = env.reset()[0]
        hx = torch.zeros(1, 256)
        score, done, steps = 0, False, 0
        while not done and steps < STEP_MAX:
            state = torch.tensor(
                np.array(obs).reshape(1, -1), dtype=torch.float)
            action, value, log_prob, hx = local_agent(state, hx)
            next_obs, reward, done, info = env.step(action)
            steps += 1
            score += reward
            reward = 0  # turn off extrinsic rewards
            memory.remember(obs, action, reward, next_obs, value, log_prob)
            obs = next_obs
            if steps % T_MAX == 0 or done:
                states, actions, rewards, new_states, values, log_probs = \
                    memory.sample_memory()
                if icm:
                    intrinsic_reward, L_I, L_F = \
                        local_icm.calc_loss(states, new_states, actions)

                loss = local_agent.calc_loss(obs, hx, done, rewards, values,
                                             log_probs, intrinsic_reward)

                optimizer.zero_grad()
                hx = hx.detach()
                if icm:
                    icm_optimizer.zero_grad()
                    (L_I + L_F).backward()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_agent.parameters(), 40)

                for local_param, global_param in zip(
                        local_agent.parameters(),
                        global_agent.parameters()):
                    global_param._grad = local_param.grad
                optimizer.step()
                local_agent.load_state_dict(global_agent.state_dict())

                if icm:
                    for local_param, global_param in zip(
                            local_icm.parameters(),
                            global_icm.parameters()):
                        global_param._grad = local_param.grad
                    icm_optimizer.step()
                    local_icm.load_state_dict(global_icm.state_dict())
                memory.clear_memory()
            # print(steps)

        scores_list.append(score)
        avg_score = np.mean(scores_list[-100:])
        if name == '1':
            print('{} : episode {:4d} | score {:6.2f} |'
                  'intrinsic_reward {:7.2f} | avg score {:5.1f}'.format(
                      algo, episode, score,
                      torch.sum(intrinsic_reward),
                      avg_score))
        episode += 1
    if name == '1':
        x = [z for z in range(max_epochs)]
        running_avg = np.zeros(len(scores_list))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores_list[max(0, i-100):(i+1)])
        plt.plot(x, running_avg)
        plt.title('Running average of previous 100 episodes')
        plt.savefig(algo + '-' + env_id+'.png')
