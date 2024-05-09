import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical


class ActorCriticBase(nn.Module):
    GAMMA = 0.99

    def __init__(
        self, num_inputs, num_actions, hidden_size, device, learning_rate=3e-4
    ):
        super(ActorCriticBase, self).__init__()
        print("input", num_inputs)
        print("action", num_actions)

        self.num_actions = num_actions
        self.device = device
        # hidden_size = round(math.sqrt(num_inputs * num_actions))
        print("hidden size:", hidden_size)
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, hidden_size)
        self.critic_linear3 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear3 = nn.Linear(hidden_size, num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # nn.init.xavier_normal_(self.critic_linear1.weight)
        # # nn.init.xavier_normal_(self.critic_linear2.weight)
        # nn.init.xavier_normal_(self.critic_linear3.weight)
        # nn.init.xavier_normal_(self.actor_linear1.weight)
        # # nn.init.xavier_normal_(self.actor_linear2.weight)
        # nn.init.xavier_normal_(self.actor_linear3.weight)

    def clear(self):
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.entropy_term = 0

    def step(self):
        Qval = torch.tensor(0.0, device=self.device)

        # compute Q values
        Qvals = torch.zeros(len(self.values)).to(self.device)
        for t in reversed(range(len(self.rewards))):
            Qval = self.rewards[t] + self.GAMMA * Qval
            Qvals[t] = Qval

        # update actor critic
        values = torch.stack(self.values)
        # Qvals = torch.tensor(Qvals)
        log_probs = torch.stack(self.log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * self.entropy_term

        self.optimizer.zero_grad()
        ac_loss.backward()
        self.optimizer.step()


class ActorCriticC9(ActorCriticBase):
    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).reshape(
            1, -1
        )
        # Critic - Value function V
        value = F.relu(self.critic_linear1(state))
        value = F.relu(self.critic_linear2(value))
        value = self.critic_linear3(value)

        # Actor - Policy
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.relu(self.actor_linear2(policy_dist))
        policy_dist_final = F.softmax(self.actor_linear3(policy_dist), dim=1)

        return value, policy_dist_final, policy_dist.max(1)[0]


class ActorCriticC14(ActorCriticBase):
    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).reshape(
            1, -1
        )
        # Critic - Value function V
        value = F.relu(self.critic_linear1(state))
        value = F.relu(self.critic_linear2(value))
        value = self.critic_linear3(value)

        # Actor - Policy
        policy_dist = F.tanh(self.actor_linear1(state))
        policy_dist = F.tanh(self.actor_linear2(policy_dist))
        policy_dist_final = F.softmax(self.actor_linear3(policy_dist), dim=1)

        return value, policy_dist_final, policy_dist.max(1)[0]


class ActorCriticC15(ActorCriticBase):
    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).reshape(
            1, -1
        )
        # Critic - Value function V
        value = F.tanh(self.critic_linear1(state))
        value = F.tanh(self.critic_linear2(value))
        value = self.critic_linear3(value)

        # Actor - Policy
        policy_dist = F.tanh(self.actor_linear1(state))
        policy_dist = F.tanh(self.actor_linear2(policy_dist))
        policy_dist_final = F.softmax(self.actor_linear3(policy_dist), dim=1)

        return value, policy_dist_final, policy_dist.max(1)[0]
