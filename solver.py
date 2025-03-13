
import random
import time

# import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

from tqdm import tqdm

from grid_env import GridEnv

class Solve:
    def __init__(self, env: GridEnv):
        self.gama = 0.9
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.reward_space_size, self.reward_list = len(self.env.reward_list), self.env.reward_list
        self.state_value = np.zeros(shape=self.state_space_size)
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))
        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()
        self.writer = SummaryWriter("logs")  # 实例化SummaryWriter对象

    def show_policy(self):
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy = self.policy[state, action]
                self.env.render_.draw_action(pos=self.env.state2pos(state),
                                             toward=policy * 0.4 * self.env.action_to_direction[action],
                                             radius=policy * 0.1)

    def show_policy(self, policy):
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy_cur = policy[state, action]
                self.env.render_.draw_action(pos=self.env.state2pos(state),
                                             toward=policy_cur * 0.4 * self.env.action_to_direction[action],
                                             radius=policy_cur * 0.1)
    
    def show_state_value(self, state_value, y_offset=0.2):
        for state in range(self.state_space_size):
            self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(state_value[state], 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)

    def get_state_value_by_policy(self, policy, epoch, gamma):
        """
            calculate p_pi
            iterate calculate v
        """
        ped = np.expand_dims(policy, axis=1)
        ps = np.matmul(ped, self.env.Psa).squeeze(1)

        policy_greedy = np.argmax(policy, axis=1)
        policy_greedy = np.expand_dims(policy_greedy, axis=1)

        self.env.reward_list = [0, 1, -1, -1]
        
        reward_immediate = [self.reward_list[np.argmax(self.env.Rsa[s, int(x)])] for s, x in enumerate(policy_greedy)]

        for i in range(epoch):
            self.state_value = reward_immediate + gamma * np.matmul(ps, self.state_value)
            
    def bellman_optimality_equation(self, epoch, gamma, policy=None):
        """
            Value Iteration
            Thinking: 1. calculate q(s,a) 2. select optimal pi -> update v(s)
        """

        self.env.reward_list = [0, 1, -10, -1]
        
        for i in range(epoch):
            qsa = np.matmul(self.env.Rsa, self.env.reward_list) + gamma * np.matmul(self.env.Psa, self.state_value)

            self.policy = np.eye(5, dtype=int)[np.argmax(qsa, axis=-1)]
            
            self.state_value = (self.policy * qsa).sum(axis=1)

    def mc_basic(self, epoch, gamma, episode=100, sample_nums=50):
        """
        
        """
        def sample_by_sa(env: GridEnv, policy, state_start, action_start, episode, sample_nums):

            rewards_all = []
            indices = np.arange(env.action_space_size)
            for _ in range(sample_nums):
                env.agent_location = state_start
                action = action_start
                rewards = []
    
                for i in range(episode):
                    _, reward, terminated, _, _ = env.step(action)

                    action_cur_probs = policy[env.pos2state(env.agent_location)]
                    action = np.random.choice(indices, p=action_cur_probs)
                    
                    rewards.append(reward)
    
                env.reset()

                rewards_all.append(rewards)
            return rewards_all

        self.env.reward_list = [0, 1, -10, -1]
        self.policy = np.eye(5, dtype=int)[[4]*self.state_space_size]
        gamma_sample_nums = gamma ** np.arange(episode)
        gamma_sample_nums = np.expand_dims(gamma_sample_nums, 0)

        for _ in tqdm(range(epoch), total=epoch):

            q_value_all = []
            # for i in tqdm(range(self.state_space_size), total=self.state_space_size, desc="states "):
            for i in range(self.state_space_size):
                state_cur = self.env.state2pos(i)

                q_value = []
                for j in range(self.action_space_size):
                    action_cur = j

                    rewards = sample_by_sa(self.env, self.policy, state_cur, action_cur, episode, sample_nums)
                    rewards_discounted = gamma_sample_nums * rewards

                    q_value.append(np.mean(rewards_discounted))
                    
                q_value_all.append(q_value)

            # print(np.argmax(q_value_all[11]), q_value_all[11])
            self.policy = np.eye(5, dtype=int)[np.argmax(q_value_all, axis=-1)]

        self.get_state_value_by_policy(self.policy, 100, gamma)

    def mc_exploring_starts(self, epoch, gamma, episode=100, sample_nums=50):
        
        def sample_by_sa(env: GridEnv, policy, state_start, action_start, episode, sample_nums):
            
            states_all = []
            actions_all = []
            rewards_all = []
            indices = np.arange(env.action_space_size)
            for _ in range(sample_nums):
                env.agent_location = state_start
                action = action_start
                rewards = []
                states = []
                actions = []

                for i in range(episode):
                    states.append(env.pos2state(env.agent_location))
                    actions.append(action)

                    _, reward, terminated, _, _ = env.step(action)

                    action_cur_probs = policy[env.pos2state(env.agent_location)]
                    action = np.random.choice(indices, p=action_cur_probs)
                    
                    rewards.append(reward)
                    
                env.reset()

                states_all.append(states)
                actions_all.append(actions)
                rewards_all.append(rewards)
            return states_all, actions_all, rewards_all
        
        def valid_sa_single(states, actions):
            hist = set()
            valid = []
            for state, actoin in zip(states, actions):
                
                if (state, actoin) not in hist:
                    valid.append(True)
                    hist.add((state, actoin))
                else:
                    valid.append(False)
            return valid

        self.env.reward_list = [0, 1, -10, -1]
        self.policy = np.eye(5, dtype=int)[[4]*self.state_space_size]
        gamma_sample_nums = gamma ** np.arange(episode)
        gamma_sample_nums = np.expand_dims(gamma_sample_nums, 0)

        action_value_env = [[-10000]*self.env.action_space_size for _ in range(self.state_space_size)]
        action_value_env_nums = [[0]*self.env.action_space_size for _ in range(self.state_space_size)]
        for _ in tqdm(range(epoch), total=epoch):

            # for i in tqdm(range(self.state_space_size), total=self.state_space_size, desc="states "):
            for i in range(self.state_space_size):
                pos_cur = self.env.state2pos(i)

                for j in range(self.action_space_size):

                    for __ in range(sample_nums):
                        action_cur = j

                        states, actions, rewards = sample_by_sa(self.env, self.policy, pos_cur, action_cur, episode, 1)
                        states, actions, rewards = states[0], actions[0], rewards[0]

                        valid_sa = valid_sa_single(states, actions)
                        g = 0

                        for state, action, reward, valid in reversed(list(zip(states, actions, rewards, valid_sa))):
                            
                            g = reward + gamma * g

                            if valid:
                                action_value_env_nums[state][action] += 1

                                if action_value_env_nums[state][action]==1:
                                    action_value_env[state][action] = g
                                else:
                                    action_value_env[state][action] = action_value_env[state][action] + (g - action_value_env[state][action])/action_value_env_nums[state][action]
                            
                            self.policy[state] = np.eye(self.env.action_space_size)[np.argmax(action_value_env[state])]
        
        self.get_state_value_by_policy(self.policy, 100, gamma)


    def mc_epsilon_greedy(self, epoch, gamma, epsilon=0.1, episode=100, sample_nums=50):
        def sample_by_sa(env: GridEnv, policy, state_start, action_start, episode, sample_nums):
            
            states_all = []
            actions_all = []
            rewards_all = []
            indices = np.arange(env.action_space_size)
            for _ in range(sample_nums):
                env.agent_location = state_start
                action = action_start
                rewards = []
                states = []
                actions = []

                for i in range(episode):
                    states.append(env.pos2state(env.agent_location))
                    actions.append(action)

                    # env.render_.draw_episode()

                    _, reward, terminated, _, _ = env.step(action)

                    action_cur_probs = policy[env.pos2state(env.agent_location)]
                    action = np.random.choice(indices, p=action_cur_probs)
                    
                    rewards.append(reward)

                env.reset()

                states_all.append(states)
                actions_all.append(actions)
                rewards_all.append(rewards)

            # env.render_.draw_episode()
            # env.render_.show_frame()
            return states_all, actions_all, rewards_all
        
        def valid_sa_single(states, actions):
            hist = set()
            valid = []
            for state, actoin in zip(states, actions):
                
                if (state, actoin) not in hist:
                    valid.append(True)
                    hist.add((state, actoin))
                else:
                    valid.append(False)
            return valid

        self.env.reward_list = [0, 1, -10, -1]

        # self.policy = np.eye(5, dtype=int)[[4]*self.state_space_size]
        gamma_sample_nums = gamma ** np.arange(episode)
        gamma_sample_nums = np.expand_dims(gamma_sample_nums, 0)

        action_value_all = np.ones(shape=(self.state_space_size, self.action_space_size)) * -10000
        
        for _ in tqdm(range(epoch), total=epoch):

            # # for i in tqdm(range(self.state_space_size), total=self.state_space_size, desc="states "):
            # for i in range(self.state_space_size):
            #     pos_cur = self.env.state2pos(i)

            #     for j in range(self.action_space_size):

            #         for __ in range(sample_nums):
            #             action_cur = j

            action_value_env = [[-10000]*self.env.action_space_size for _ in range(self.state_space_size)]
            action_value_env_nums = [[0]*self.env.action_space_size for _ in range(self.state_space_size)]

            state_sample_start = np.random.choice(self.state_space_size)
            action_sample_start = np.random.choice(self.action_space_size)

            states, actions, rewards = sample_by_sa(self.env, self.policy, self.env.state2pos(state_sample_start), action_sample_start, episode, 1)
            states, actions, rewards = states[0], actions[0], rewards[0]

            g = 0
            for state, action, reward in reversed(list(zip(states, actions, rewards))):
                
                g = reward + gamma * g

                action_value_env_nums[state][action] += 1

                if action_value_env_nums[state][action]==1:
                    action_value_env[state][action] = g
                else:
                    action_value_env[state][action] = action_value_env[state][action] + (g - action_value_env[state][action])/action_value_env_nums[state][action]
                
                action_value_all = np.maximum(action_value_all, action_value_env).tolist()

                self.policy[state] = [round(epsilon/self.env.action_space_size, 8)] * self.env.action_space_size
                self.policy[state][np.argmax(action_value_all[state])] = 1 - self.policy[state][0]*(self.env.action_space_size-1)

            # action_value_all = np.maximum(action_value_all, action_value_env).tolist()
            # self.policy = [round(epsilon/self.env.action_space_size, 8)] * self.env.action_space_size
            # self.policy[:][np.argmax(action_value_all, axis=-1).tolist()] = 1 - round(epsilon/self.env.action_space_size, 8)*(self.env.action_space_size-1)

        # policy_greedy = np.eye(self.action_space_size, dtype=int)[np.argmax(self.policy, axis=-1)]
        
        # self.get_state_value_by_policy(self.policy, 100, gamma)

        self.get_state_value_by_policy(np.eye(self.action_space_size, dtype=int)[np.argmax(self.policy, axis=-1)], 100, gamma)


    def td_sarsa(self, epoch, gamma, alpha=0.1, epsilon=0.1, episode_nums=100, sample_nums=50):
        
        def sample_by_sa(env: GridEnv, policy, state_start, action_start, episode, sample_nums):
            
            states_all = []
            actions_all = []
            rewards_all = []
            indices = np.arange(env.action_space_size)
            for _ in range(sample_nums):
                env.agent_location = state_start
                action = action_start
                rewards = []
                states = []
                actions = []

                for i in range(episode):
                    states.append(env.pos2state(env.agent_location))
                    actions.append(action)

                    # env.render_.draw_episode()

                    _, reward, terminated, _, _ = env.step(action)

                    action_cur_probs = policy[env.pos2state(env.agent_location)]
                    action = np.random.choice(indices, p=action_cur_probs)
                    
                    rewards.append(reward)

                    if terminated:
                        states.append(env.pos2state(env.agent_location))
                        actions.append(action)
                        _, reward, terminated, _, _ = env.step(action)
                        rewards.append(reward)
                        break

                env.reset()

                states_all.append(states)
                actions_all.append(actions)
                rewards_all.append(rewards)

            # env.render_.draw_episode()
            # env.render_.show_frame()
            return states_all, actions_all, rewards_all

        def exp2episode(states, actions, rewards):
            episode = []
            for i in range(len(states)-1):
                episode.append((states[i], actions[i], rewards[i], states[i+1], actions[i+1]))
            return episode

        self.env.reward_list = [0, 1, -10, -1]

        # self.policy = np.eye(5, dtype=int)[[4]*self.state_space_size]
        gamma_sample_nums = gamma ** np.arange(episode_nums)
        gamma_sample_nums = np.expand_dims(gamma_sample_nums, 0)

        action_value_all = np.ones(shape=(self.state_space_size, self.action_space_size)) * -10000

        for _ in tqdm(range(epoch), total=epoch):

            # # for i in tqdm(range(self.state_space_size), total=self.state_space_size, desc="states "):
            # for i in range(self.state_space_size):
            #     pos_cur = self.env.state2pos(i)

            #     for j in range(self.action_space_size):

            #         for __ in range(sample_nums):
            #             action_cur = j

            q_value = np.zeros(shape=(self.state_space_size, self.action_space_size))
            action_value_env = [[-10000]*self.env.action_space_size for _ in range(self.state_space_size)]
            action_value_env_nums = [[0]*self.env.action_space_size for _ in range(self.state_space_size)]

            states, actions, rewards = sample_by_sa(self.env, self.policy, self.env.state2pos(0), 0, episode_nums, 1)
            states, actions, rewards = states[0], actions[0], rewards[0]
            
            episode = exp2episode(states, actions, rewards)

            for state, action, reward, state_next, action_next in episode:
                td_target = reward + gamma * q_value[state_next][action_next]
                td_error =  q_value[state][action] - td_target
                q_value[state][action] = q_value[state][action] - alpha * td_error
            
            action_value_all = np.maximum(action_value_all, q_value).tolist()

            self.policy = np.full((self.state_space_size, self.env.action_space_size), round(epsilon/self.env.action_space_size, 8))
            for state in range(self.state_space_size):
                self.policy[state][np.argmax(action_value_all[state])] = 1 - round(epsilon/self.env.action_space_size, 8)*(self.env.action_space_size-1)
        
        self.get_state_value_by_policy(self.policy, 100, gamma)


if __name__ == "__main__":
    
    grid_env = GridEnv(size=5, target=[2, 3],
             forbidden=[[2, 2], [2, 1], [1, 1], [3, 3], [1, 3], [1, 4]],
             render_mode='')

    solver = Solve(grid_env)

    # solver.mc_basic(
    #     epoch=50, 
    #     gamma=0.9,
    #     sample_nums=10
    # )

    # solver.mc_exploring_starts(
    #     epoch=10, 
    #     gamma=0.9,
    #     sample_nums=1
    # )
    
    """
        随机采样过长之后，导致稳定性变差
        需要设置全局 qvalue，获取不同 policy 的最大 qvalue，以此更新 policy
        还是有点bug
    """
    # solver.mc_epsilon_greedy(
    #     epsilon=0.1,
    #     epoch=1000,
    #     episode=1000,
    #     gamma=0.7,
    #     sample_nums=1
    # )

    solver.td_sarsa(
        alpha=0.1,
        epsilon=0.8,
        epoch=10000,
        episode_nums=10000,
        gamma=0.9,
        sample_nums=1
    )

    solver.show_policy(solver.policy)
    solver.show_state_value(solver.state_value)

    solver.env.render()