import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SharedNetworkA2C(nn.Module):
    def __init__(self, num_actions, input_shape=(4, 84, 84)):
        super(SharedNetworkA2C, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Linear(conv_out_size, 512)
        
        # Actor
        self.actor = nn.Linear(512, num_actions)
        
        # Critic
        self.critic = nn.Linear(512, 1)

    def _get_conv_out(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        
        policy_logits = self.actor(x)
        value = self.critic(x)
        
        return policy_logits, value

class A2CAgent:
    def __init__(self, num_actions, input_shape, device, hyperparameters):
        self.device = device
        self.network = SharedNetworkA2C(num_actions, input_shape).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=hyperparameters.learning_rate)
        self.num_actions = num_actions
        self.hp = hyperparameters

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy_logits, _ = self.network(state)
            action_probs = F.softmax(policy_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()
        return action

    def compute_returns_and_advantages(self, rewards, values, dones):
        returns = []
        advantages = []
        next_value = 0
        next_advantage = 0
        
        for reward, value, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            returns.insert(0, reward + self.hp.discount_factor * next_value * (1 - done))
            td_error = reward + self.hp.discount_factor * next_value * (1 - done) - value
            advantages.insert(0, td_error + self.hp.discount_factor * self.hp.gae_lambda * next_advantage * (1 - done))
            next_value = value
            next_advantage = advantages[0]
        
        return returns, advantages

    def update(self, states, actions, returns, advantages):
        # Convert lists to numpy arrays first
        states = np.array(states)
        actions = np.array(actions)
        returns = np.array(returns)
        advantages = np.array(advantages)

        # Then convert numpy arrays to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        policy_logits, values = self.network(states)
        
        # Compute policy loss
        action_log_probs = F.log_softmax(policy_logits, dim=-1)
        action_log_probs = action_log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        policy_loss = -(action_log_probs * advantages).mean()
        
        # Compute value loss
        value_loss = F.mse_loss(values.squeeze(-1), returns)
        
        # Compute entropy bonus
        entropy = -(F.softmax(policy_logits, dim=-1) * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
        
        # Compute total loss
        loss = policy_loss + self.hp.value_loss_coef * value_loss - self.hp.entropy_coef * entropy
        
        # Perform optimization step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.hp.max_grad_norm)
        self.optimizer.step()
        
        return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path))