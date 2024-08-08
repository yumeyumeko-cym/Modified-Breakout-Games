class Hyperparameters():
    def __init__(self):
        self.RL_load_path = f'./final_weights.pth'
        self.save_path = f'./final_weights'
        self.learning_rate = 3e-4
        self.discount_factor = 0.99
        self.batch_size = 32
        self.targetDQN_update_rate = 1000
        self.num_episodes = 1000
        self.num_test_episodes = 10
        self.epsilon_decay = 0.995
        self.buffer_size = 100000
        self.uncertainty_prob = 0.1
        self.frame_stack = 4
        self.update_frequency = 5
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.1
        

    def change(self, **kwargs):
        '''
        Change the hyperparameters
        '''
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid hyperparameter")
