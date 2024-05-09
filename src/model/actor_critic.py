import argparse
import json
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import random
import math
from datetime import datetime

from src.environment.action import Action
from src.components import Subnet, Device
from .model import *


from src.components import Network
from src.constants import Constants

# from src.environment import MitreEnv, MitreEnvVector, MitreEnvCluster

EPS_START = 1.0
EPS_END = 0.1


class TrainerOrigin:
    def __init__(self, env, model_class, hidden_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.hidden_size = hidden_size
        self.agent = model_class(
            num_inputs=len(env.reset()),
            num_actions=env.action_space.n,
            hidden_size=hidden_size,
            device=self.device,
        ).to(self.device)

        # print('Number of hosts: ', len(env.dict_hosts))
        print("Run in", self.device)

    def train(self, num_episodes, fpath, threshold=None):
        env = self.env
        device = self.device

        num_inputs = len(env.reset())
        num_outputs = env.action_space.n
        num_steps = 100_000_000  # Number step extremely large
        print("Maximum steps", num_steps)
        # print(len(env.dict_hosts))
        print("Run in", device)

        actor_critic = self.agent

        all_rewards = []
        all_sensitives = []
        entropy_term = 0

        # print('Has sensitive', self.env.has_sensitive)
        # sensitive_str = 't' if self.env.has_sensitive else 'f'
        # print('Has firewall', self.env.has_firewall)
        # firewall_str = 't' if self.env.has_firewall else 'f'
        # print('Has service score', self.env.has_service_score)
        # service_score_str = 't' if self.env.has_service_score else 'f'
        # fpath = f'results/actor_critic_o_{scenario}_{sensitive_str}{firewall_str}{service_score_str}_{num_episodes}_{num_steps}_{self.agent.__class__.__name__[11:].lower()}'
        # fpath = f'results/actor_critic_o'
        
        save_result = dict(
            episode=list(), reward=list(), step=list(), sensitive=list(), compromised=list(), time=list()
        )
        save_histories = list()

        if not os.path.exists(fpath):
            os.makedirs(fpath)

        for episode in range(num_episodes):
            start = datetime.now()

            log_probs = []
            values = []
            rewards = []

            state = env.reset()
            eps_threshold = (
                threshold
                if threshold is not None
                else EPS_START
                - min(episode, num_episodes // 2)
                * (EPS_START - EPS_END)
                / (num_episodes // 2)
            )
            actor_critic.clear()
            rc = 0
            scan_subnet_count = 0
            scan_host_count = 0
            exploit_host_count = 0
            scan_subnet_reward = 0
            scan_host_reward = 0
            exploit_host_reward = 0
            sensitive_count = 0
            defender_count = 0
            compromised_count = 0
            for steps in range(num_steps):
                # value: tensor shape (1 x 1)
                # policy_dist: tensor shape (1 x num_outputs)
                value, policy_dist, _ = actor_critic.forward(state)
                value = value.detach()[0, 0]
                policy_dist = Categorical(policy_dist)

                sample = random.random()
                if sample > eps_threshold:
                    action = policy_dist.sample().squeeze(0).detach()
                else:
                    proto_action = policy_dist.sample().squeeze(0).detach()
                    knn_actions = env.get_knn_actions(proto_action)
                    max_reward = -10000000
                    max_idx = -1
                    for idx, knn_action in enumerate(knn_actions):
                        _, reward, _, _ = env.step(knn_action, state_changed=False)
                        if reward > max_reward:
                            max_reward = reward
                            max_idx = idx
                    action = torch.tensor(knn_actions[max_idx], device=device)

                log_prob = policy_dist.log_prob(action).squeeze(0)
                entropy = policy_dist.entropy().mean().detach()

                new_state, reward, done, info = env.step(action)  # , pre_state=state)

                if env.action_space.get_action(action).type == Subnet:
                    scan_subnet_count += 1
                    scan_subnet_reward += reward
                # if env.action_space.get_action(action).type == Action.TYPE_SCAN_HOST:
                #     scan_host_count += 1
                #     scan_host_reward += reward
                if env.action_space.get_action(action).type == Device:
                    exploit_host_count += 1
                    exploit_host_reward += reward

                if info["exploit_sensitive"]:
                    sensitive_count += 1
                if info["reason"] == Constants.ATTACK_DEVICE_MSG.SUCCESSFUL_COMPROMISED:
                    compromised_count += 1

                actor_critic.rewards.append(reward)
                actor_critic.values.append(value)
                actor_critic.log_probs.append(log_prob)
                actor_critic.entropy_term += entropy
                state = new_state

                if done or steps == num_steps - 1:
                    # Qval, _ = actor_critic.forward(state)
                    # Qval = Qval.detach()[0,0]
                    Qval = torch.tensor(0.0, device=device)
                    break

            all_rewards.append(np.sum(actor_critic.rewards))
            all_sensitives.append(sensitive_count)
            average_reward = np.mean(all_rewards[-10:])
            sys.stdout.write(
                "episode: {}, reward: {}, total length: {}, average rewards: {}, [{}, {}], e: {} \n".format(
                    episode,
                    all_rewards[-1:],
                    steps,
                    average_reward,
                    sensitive_count,
                    defender_count,
                    eps_threshold,
                )
            )
            print(
                "{:7} {:7} {:7} - {:7} {:7} {:7}\n".format(
                    scan_subnet_count,
                    scan_host_count,
                    exploit_host_count,
                    scan_subnet_reward,
                    scan_host_reward,
                    exploit_host_reward,
                )
            )
            # compute Q values
            actor_critic.step()

            save_histories.append(env.history.to_json())

            save_result["episode"].append(episode)
            save_result["reward"].append(np.sum(actor_critic.rewards))
            save_result["step"].append(steps)
            save_result["sensitive"].append(sensitive_count)
            save_result["compromised"].append(compromised_count)
            save_result["time"].append((datetime.now() - start).total_seconds() * 1000)

            if (episode + 1) % 100 == 0:
                print("SAVING", episode + 1)
                pd.DataFrame.from_dict(save_result).to_csv(
                    f"{fpath}/save_result.csv", index=False
                )

                with open(f"{fpath}/history.json", "w") as outfile:
                    json.dump(save_histories, outfile)

        pd.DataFrame.from_dict(save_result).to_csv(
            f"{fpath}/save_result.csv", index=False
        )

        with open(f"{fpath}/history.json", "w") as outfile:
            json.dump(save_histories, outfile)

        torch.save(actor_critic.state_dict(), fpath + "/actor_critic.pt")

        ##### Plot results
        # smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
        # smoothed_rewards = [elem for elem in smoothed_rewards]
        # plt.plot(all_rewards)
        # plt.plot(smoothed_rewards)
        # plt.plot()
        # plt.xlabel('Episode')
        # plt.ylabel('Reward')
        # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--threshold", help="threshold (0:a2c, 1:wa)", default=None, type=float
    )
    parser.add_argument("-s", "--scenario", help="scenario (1-4)", default="16_2_5")
    parser.add_argument(
        "-e", "--episode", help="number of episodes", default=1000, type=int
    )
    parser.add_argument(
        "-d",
        "--difficulty",
        help="difficulty (easy, medium, hard, extreme)",
        default="easy",
    )
    parser.add_argument("-k", "--knn", help="knn rate", default=0.01, type=float)
    args = parser.parse_args()
    print(args)

    reader = ScenarioReader()
    # mitigation_info = reader.read_mitigation_info("data/mitigation_info.csv")
    json_file = open(f"mitre/scenario/{args.scenario}.json")
    data = json.load(json_file)
    net = Network()
    net.initialize(data).set_foothold("External", "Attacker")
    # defender = DefenderAgent(net, mitigation_info)
    env = MitreEnvCluster(
        net,  # Choose between MitreEnvVector and MitreEnvCluster
        technique_info=reader.read_technique_info(
            f"mitre/technique/technique_{args.difficulty}.csv"
        ),
        tatic_emb_path="mitre/tatic_onehot.pickle",
        technique_emb_path="mitre/technique_weight.pickle",
        knn_rate=args.knn,
    )
    # reader = DataReader("data/" + scenario + ".json")
    # reader.read()
    # env = MdpEnv(reader.dict_subnets, reader.dict_services, ['192.168.0.0'], has_unknown_state=True, has_sensitive=True, has_service_score=True)

    agent = TrainerOrigin(env, model_class=ActorCriticC15, hidden_size=64)
    agent.train(
        num_episodes=args.episode,
        fpath=f"results/cluster-a2c-{args.threshold}-{args.difficulty}-{args.scenario}-{args.knn}",
        threshold=args.threshold,
    )
