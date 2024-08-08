import torch
import pygame
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from Hyperparameters import Hyperparameters
from Agent import Agent
import cv2
from tqdm import tqdm


class ActionUncertaintyWrapper(gym.Wrapper):
    def __init__(self, env, uncertainty_prob=0.01):
        super().__init__(env)
        self.uncertainty_prob = uncertainty_prob
        self.action_space = env.action_space

        
    def step(self, action):
        if np.random.random() < self.uncertainty_prob:
            action = self.action_space.sample()
        return self.env.step(action)
    '''
    def step(self, action):
        original_action = action
        if np.random.random() < self.uncertainty_prob:
            action = self.action_space.sample()
        observation, reward, terminated, truncated, info = self.env.step(action)
        info['action_changed'] = (original_action != action)
        info['original_action'] = original_action
        info['actual_action'] = action
        return observation, reward, terminated, truncated, info
    '''
class DQL():
    def __init__(self, hyperparameters:Hyperparameters, train_mode):

        if train_mode:
            render = None
        else:
            render = "rgb_array"
            #render = "human"

        # Attention: <self.hp> contains all hyperparameters that you need
        # Checkout the Hyperparameter Class
        self.hp = hyperparameters

        # Load the environment
        self.env = gym.make('Breakout-v4', render_mode=render)
        self.env = ActionUncertaintyWrapper(self.env, uncertainty_prob=self.hp.uncertainty_prob)


        # Initiate the Agent
        self.agent = Agent(env = self.env, hyperparameters = self.hp)

        self.frame_buffer = np.zeros((4, 84, 84), dtype=np.uint8)

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized
        

    def feature_representation(self, state:int):
        """
        Preprocess the state from Breakout
        """

        # Shift frames in the buffer
        processed_frame = self.preprocess_frame(state)
        self.frame_buffer[:-1] = self.frame_buffer[1:]
        self.frame_buffer[-1] = processed_frame
        # Convert to float and normalize
        state = np.array(self.frame_buffer, dtype=np.float32) / 255.0
        return torch.FloatTensor(state)
        # shape = (1, 4, 84, 84)
    
    def train(self): 
        """                
        Training the DQN via DQL
        """
        
        total_steps = 0
        self.collected_rewards = []
        
        # Training loop
        for episode in tqdm(range(1, self.hp.num_episodes+1), desc="Training"):
            state, _ = self.env.reset()
            state = self.feature_representation(state)
            ended = False
            truncated = False
            episode_reward = 0
                                                
            while not ended and not truncated:
                action = self.agent.epsilon_greedy(state)
                next_state, reward, ended, truncated, _ = self.env.step(action)
                next_state = self.feature_representation(next_state)

                # Put it into replay buffer
                self.agent.replay_buffer.store(state, action, next_state, reward, ended) 
                
                if len(self.agent.replay_buffer) > self.hp.batch_size:
                    if total_steps % self.hp.update_frequency == 0:
                        self.agent.apply_SGD(ended)
                    if total_steps % self.hp.targetDQN_update_rate == 0:
                        self.agent.update_target()

                state = next_state
                episode_reward += reward
                total_steps += 1
                            
            self.collected_rewards.append(episode_reward)                     
                                                                        
            # Decay epsilon at the end of each episode
            self.agent.epsilon = max(self.hp.final_epsilon, self.agent.epsilon * self.hp.epsilon_decay) 

            # Update tqdm postfix with latest information
            if episode % 100 == 0 or episode == self.hp.num_episodes:
                avg_reward = np.mean(self.collected_rewards[-100:])
                tqdm.write(f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.agent.epsilon:.2f}")
            

        self.agent.save(self.hp.save_path + '.pth')
        self.plot_learning_curves()
                                                                    

    def play(self):  
        """                
        play with the learned policy
        You can only run it if you already have trained the DQN and saved its weights as .pth file
        """
           
        # Load the trained DQN
        self.agent.onlineDQN.load_state_dict(torch.load(self.hp.RL_load_path, map_location=torch.device(self.agent.device)))
        self.agent.onlineDQN.eval()
        
        # Playing 
        for episode in range(1, self.hp.num_test_episodes+1):         
            state, _ = self.env.reset()
            ended = False
            truncated = False
            step_size = 0
            episode_reward = 0
                                                           
            while not ended and not truncated:         
                state_feature = self.feature_representation(state)
                action = self.agent.greedy(state_feature)
                
                next_state, reward, ended, truncated, _ = self.env.step(action)
                                
                state = next_state
                episode_reward += reward
                step_size += 1
                                                                                                                       
            # Print Results of Episode            
            printout = (f"Episode: {episode}, "
                      f"Steps: {step_size:}, "
                      f"Sum Reward of Episode: {episode_reward:.2f}, ")
            print(printout)
            
        pygame.quit()







    ############## THIS METHOD HAS BEEN COMPLETED AND YOU DON'T NEED TO MODIFY IT ################
    def plot_learning_curves(self):
        # Calculate the Moving Average over last 100 episodes
        moving_average = np.convolve(self.collected_rewards, np.ones(100)/100, mode='valid')
        
        plt.figure()
        plt.title("Reward")
        plt.plot(self.collected_rewards, label='Reward', color='gray')
        plt.plot(moving_average, label='Moving Average', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        
        # Save the figure
        plt.savefig(f'./Reward_vs_Episode_uncertainty={self.hp.uncertainty_prob}.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close() 
        
                
        plt.figure()
        plt.title("Loss")
        plt.plot(self.agent.loss_list, label='Loss', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Training Loss")
        
       # Save the figure
        plt.savefig(f'./Learning_Curve_uncertainty={self.hp.uncertainty_prob}.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()        
        

