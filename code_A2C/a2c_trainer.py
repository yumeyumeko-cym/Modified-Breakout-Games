import torch
import numpy as np
import gymnasium as gym
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from a2c_agent import A2CAgent

from Hyperparameters import Hyperparameters

class ActionUncertaintyWrapper(gym.Wrapper):
    def __init__(self, env, uncertainty_prob=0.01):
        super().__init__(env)
        self.uncertainty_prob = uncertainty_prob
        self.action_space = env.action_space

    def step(self, action):
        if np.random.random() < self.uncertainty_prob:
            action = self.action_space.sample()
        return self.env.step(action)


class A2CTrainer:
    def __init__(self, hyperparameters:Hyperparameters, train_mode=True):
        self.hp = hyperparameters
        self.train_mode = train_mode
        
        # Set up the environment
        render_mode = None if train_mode else "rgb_array"
        self.env = gym.make('Breakout-v4', render_mode=render_mode)
        self.env = ActionUncertaintyWrapper(self.env, uncertainty_prob=self.hp.uncertainty_prob)


        # Set up the agent
        num_actions = self.env.action_space.n
        input_shape = (4, 84, 84)  # stacking 4 frames
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = A2CAgent(num_actions, input_shape, device, self.hp)
        
        self.frame_buffer = np.zeros((4, 84, 84), dtype=np.uint8)

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def stack_frames(self, state):
        processed_frame = self.preprocess_frame(state)
        self.frame_buffer[:-1] = self.frame_buffer[1:]
        self.frame_buffer[-1] = processed_frame
        return np.array(self.frame_buffer, dtype=np.float32) / 255.0

    def train(self):
        total_steps = 0
        episode_rewards = []
        
        for episode in tqdm(range(1, self.hp.num_episodes + 1), desc="Training"):
            state, _ = self.env.reset()
            state = self.stack_frames(state)
            episode_reward = 0
            done = False
            
            states, actions, rewards, values, dones = [], [], [], [], []
            
            for step in range(self.hp.max_steps_per_episode):
                action = self.agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.stack_frames(next_state)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(self.agent.network(torch.FloatTensor(state).unsqueeze(0).to(self.agent.device))[1].item())
                dones.append(done)
                
                state = next_state
                episode_reward += reward
                total_steps += 1
                
                if done or step == self.hp.max_steps_per_episode - 1:
                    next_value = 0 if done else self.agent.network(torch.FloatTensor(next_state).unsqueeze(0).to(self.agent.device))[1].item()
                    returns, advantages = self.agent.compute_returns_and_advantages(rewards, values + [next_value], dones)
                    loss, policy_loss, value_loss, entropy = self.agent.update(states, actions, returns, advantages)
                    
                    episode_rewards.append(episode_reward)
                    
                    if episode % 100 == 0:
                        avg_reward = np.mean(episode_rewards[-100:])
                        tqdm.write(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")
                    
                    break
            

                
        self.agent.save(self.hp.save_path + '.pth')
        self.plot_learning_curve(episode_rewards)

    def play(self):
        self.agent.load(self.hp.load_path)
        for episode in range(self.hp.num_test_episodes):
            state, _ = self.env.reset()
            state = self.stack_frames(state)
            episode_reward = 0
            done = False
            
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.stack_frames(next_state)
                
                state = next_state
                episode_reward += reward
            
            print(f"Test Episode {episode + 1}, Reward: {episode_reward}")

    def plot_learning_curve(self, rewards):
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.title('Learning Curve')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig('learning_curve.png')
        plt.close()