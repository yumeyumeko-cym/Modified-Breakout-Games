from Hyperparameters import Hyperparameters
from DQL import DQL
import torch

if __name__ == '__main__':
    hyperparameters = Hyperparameters()
    
    # change hyperparameters
    hyperparameters.change(
        num_episodes = 1000,
        #learning_rate = 5e-4,
        #num_test_episodes = 10,
        #uncertainty_prob = 0.1
        #final_epsilon = 0.1
        #update_frequency = 10
        discount_factor = 0.50
    )



    train = True
    #train = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    DRL = DQL(hyperparameters, train_mode=train)

    if train:
        print("Starting training...")
        DRL.train()
        print("Training completed.")
    else:
        print("Starting evaluation...")
        DRL.play()
        print("Evaluation completed.")