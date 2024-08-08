class Hyperparameters:
    def __init__(self):
        # General parameters
        self.num_episodes = 1000
        self.max_steps_per_episode = 10000
        self.save_interval = 500
        self.num_test_episodes = 10

        # A2C specific parameters
        self.learning_rate = 3e-4
        self.discount_factor = 0.99
        self.gae_lambda = 0.95
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

        


        # Environment parameters
        self.frame_stack = 4
        self.uncertainty_prob = 0.1

        # Paths
        self.save_path = './a2c_model_final'
        self.load_path = './a2c_model_final.pth'

    def change(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid hyperparameter")