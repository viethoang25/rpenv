import json
import os
from datetime import datetime
from src.components import Network
from src.constants import Constants

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from PIL import Image
import pickle
from collections import namedtuple
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, inputs, outputs, nodes=128):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(inputs, nodes)
        self.fc2 = nn.Linear(nodes, nodes)
        self.fc3 = nn.Linear(nodes, outputs)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class DuelingDQN(nn.Module):
    def __init__(self, inputs, outputs):
        super(DuelingDQN, self).__init__()

        self.feature = nn.Sequential(nn.Linear(inputs, 512), nn.Tanh())

        self.advantage = nn.Sequential(
            nn.Linear(512, 512), nn.Tanh(), nn.Linear(512, outputs)
        )

        self.value = nn.Sequential(nn.Linear(512, 512), nn.Tanh(), nn.Linear(512, 1))

    def forward(self, x):
        feature = self.feature(x)
        advantage = self.advantage(feature)
        value = self.value(feature)
        return value + advantage - advantage.mean()


class Agent:
    ALGO_DQN = "dqn"
    ALGO_DDQN = "ddqn"
    ALGO_DUEL_DQN = "dueling_dqn"
    ALGO_D3QN = "d3qn"

    def __init__(
        self,
        env,
        memory_capacity=1000000,
        batch_size=50,
        target_update=10,
        algorithm=ALGO_DQN,
        is_two_steps=False,
        mode_two_steps=None,
        num_nodes=128,
    ):
        self.is_ipython = "inline" in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        self.GAMMA = 0.999
        self.BATCH_SIZE = batch_size
        self.EPS_START = 1.0
        self.EPS_END = 0.05
        self.TARGET_UPDATE = target_update

        self.num_episodes = 1000
        self.eps_decay_episodes = self.num_episodes
        self.algorithm = algorithm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.env = env
        if not is_two_steps:
            self.env_size = len(self.env.reset())
            self.n_actions = self.env.action_space.n
        else:
            self.env_size = len(self.env.reset(mode_two_steps))
            self.n_actions = self.env.action_space.n(mode_two_steps)

        if self.algorithm == self.ALGO_DQN or self.algorithm == self.ALGO_DDQN:
            print("DQN")
            self.policy_net = DQN(self.env_size, self.n_actions, num_nodes).to(
                self.device
            )
            self.target_net = DQN(self.env_size, self.n_actions, num_nodes).to(
                self.device
            )
        else:
            print("Dueling DQN")
            self.policy_net = DuelingDQN(self.env_size, self.n_actions).to(self.device)
            self.target_net = DuelingDQN(self.env_size, self.n_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(memory_capacity)

        self.eps_threshold = self.EPS_START

        self.episode_rewards = []
        plt.ion()

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        plt.title("Training...")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        base_averages = 50
        if len(durations_t) >= base_averages:
            means = durations_t.unfold(0, base_averages, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(base_averages - 1), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def select_action(self, state, cur_episode=1):
        sample = random.random()
        self.eps_threshold = (
            self.EPS_START
            - min(cur_episode, self.eps_decay_episodes)
            * (self.EPS_START - self.EPS_END)
            / self.eps_decay_episodes
        )
        if sample > self.eps_threshold:
            with torch.no_grad():
                return (
                    self.policy_net(state.view(-1, self.env_size)).max(1)[1].view(1, 1)
                )
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=self.device,
                dtype=torch.long,
            )

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # print(batch.next_state)
        next_state_batch = torch.cat(batch.next_state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch
        )  # Q(s,a)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        if self.algorithm == self.ALGO_DQN or self.algorithm == self.ALGO_DUEL_DQN:
            # DQN
            next_state_values = (
                self.target_net(next_state_batch).max(1)[0].detach()
            )  # max Q'(s+1,a)
        elif self.algorithm == self.ALGO_DDQN or self.algorithm == self.ALGO_D3QN:
            # Double DQN
            next_action_indices = (
                self.policy_net(next_state_batch).max(1)[1].detach()
            )  # argmax(Q(s,a))
            next_action_indices = next_action_indices.to(self.device).reshape(-1, 1)
            next_state_values = (
                self.target_net(next_state_batch)
                .gather(1, next_action_indices)
                .detach()
            )  # Q'(s+1,a)
            next_state_values = next_state_values.reshape(-1)

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        # print(next_state_values.shape, expected_state_action_values.shape)

        # Compute Huber loss
        loss = F.mse_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def init_memory(self):
        while len(self.memory) < self.BATCH_SIZE:
            state = (
                torch.tensor(self.env.reset(), device=self.device)
                .float()
                .view(1, self.env_size)
            )
            done = False
            while not done:
                action = self.env.action_space.sample()
                current_state, reward, done, _ = self.env.step(action)
                current_state = (
                    torch.tensor(current_state, device=self.device)
                    .float()
                    .view(1, self.env_size)
                )
                action = torch.tensor([[action]], device=self.device, dtype=torch.long)
                reward = torch.tensor([reward], device=self.device)
                self.memory.push(state, action, current_state, reward)
                state = current_state
        print("done initializing memory")

    def train(self, num_episodes, model_path="policy_net.pt", eps_decay_episodes=1000):
        self.num_episodes = num_episodes
        self.eps_decay_episodes = eps_decay_episodes
        self.env.reset()
        self.init_memory()
        max_reward = -float("inf")
        num_steps = (
            2 * self.env.action_space.n
        )  # len(self.env.dict_subnets) + 4 * len(self.env.dict_hosts)
        print("Num steps", num_steps)
        save_result = dict(
            episode=list(), reward=list(), step=list(), sensitive=list(), compromised=list(), time=list()
        )

        for i_episode in range(self.num_episodes):
            start = datetime.now()
            # Initialize the environment and state
            state = (
                torch.tensor(self.env.reset(), device=self.device)
                .float()
                .view(1, self.env_size)
            )
            total_reward = 0
            total_loss = 0
            sensitive_count = 0
            compromised_count = 0
            for t in count():
                # env.render()
                # Select and perform an action
                action = self.select_action(state, i_episode)
                current_state, reward, done, info = self.env.step(
                    action.item()
                )  # , pre_state=state.view(-1).cpu().numpy().tolist())
                current_state = (
                    torch.tensor(current_state, device=self.device)
                    .float()
                    .view(1, self.env_size)
                )
                total_reward += reward
                reward = torch.tensor([reward], device=self.device)

                if info["exploit_sensitive"]:
                    sensitive_count += 1
                if info["reason"] == Constants.ATTACK_DEVICE_MSG.SUCCESSFUL_COMPROMISED:
                    compromised_count += 1

                # Store the transition in memory
                self.memory.push(state, action, current_state, reward)

                # Move to the next state
                state = current_state

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if t >= num_steps:  # 1000: #350:
                    done = True
                if done:
                    print(i_episode, total_reward, sensitive_count)
                    self.episode_rewards.append(total_reward)
                    # self.plot_durations()
                    break

            max_reward = max(max_reward, total_reward)
            
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                print("SYNC")
                self.target_net.load_state_dict(self.policy_net.state_dict())

            save_result["episode"].append(i_episode)
            save_result["reward"].append(total_reward)
            save_result["step"].append(t)
            save_result["sensitive"].append(sensitive_count)
            save_result["compromised"].append(compromised_count)
            save_result["time"].append((datetime.now() - start).total_seconds() * 1000)

            if i_episode % 10 == 0:
                print(
                    "Episode",
                    self.eps_threshold,
                    np.mean(save_result["sensitive"][-10:]),
                )

        print("Complete")
        print("Max reward", max_reward)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        ##### PLOT
        # self.plot_durations()
        torch.save(self.policy_net.state_dict(), model_path + "/policy_net.pt")
        # for param in policy_net.parameters():
        #     print(param.data)
        pd.DataFrame.from_dict(save_result).to_csv(
            f"{model_path}/save_result.csv", index=False
        )

        plt.ioff()
        plt.show()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-a", "--algorithm", help="algorithm used for model (dqn, ddqn, dueling_dqn, d3qn)", default='dqn')
#     parser.add_argument("-s", "--scenario", help="scenario (1-4)", default='16_2_5')
#     parser.add_argument("-e", "--episode", help="number of episodes", default=1000, type=int)
#     parser.add_argument("-d", "--difficulty", help="difficulty (easy, medium, hard, extreme)", default='easy')
#     args = parser.parse_args()

#     reader = ScenarioReader()
#     # mitigation_info = reader.read_mitigation_info("data/mitigation_info.csv")
#     technique_info = reader.read_technique_info(f"mitre/technique/technique_{args.difficulty}.csv")
#     json_file = open(f'mitre/scenario/{args.scenario}.json')
#     data = json.load(json_file)
#     net = Network()
#     net.initialize(data).set_foothold('External', 'Attacker')
#     env = MitreEnv(net, technique_info)

#     # reader = DataReader("data/" + scenario + ".json")
#     # reader.read()
#     # env = MdpEnv(reader.dict_subnets, reader.dict_services, ['192.168.0.0'], has_unknown_state=True, has_sensitive=True, has_service_score=True)

#     agent = Agent(env, algorithm=args.algorithm, target_update=50, num_nodes=256)
#     agent.train(num_episodes=args.episode, model_path=f'results/{args.algorithm}-{args.difficulty}-{args.scenario}', eps_decay_episodes=950)
