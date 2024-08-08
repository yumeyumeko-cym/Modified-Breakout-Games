from Hyperparameters import Hyperparameters
from a2c_trainer import A2CTrainer
import torch

if __name__ == '__main__':
    hyperparameters = Hyperparameters()
    
    hyperparameters.change(
        num_episodes=2000,
        learning_rate=3e-4,

    )


    #train_mode = True
    train_mode = False

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the A2C trainer
    trainer = A2CTrainer(hyperparameters, train_mode=train_mode)

    if train_mode:
        print("Starting training...")
        trainer.train()
        print("Training completed.")
    else:
        print("Starting evaluation...")
        trainer.play()
        print("Evaluation completed.")