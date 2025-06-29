import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.optim as optim
import pickle
import pandas as pd
import copy
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np

# ----------------------
# CUSTOM FUNCTIONS
# ----------------------
class FeatureScaler():
    # Normalizes each column (i.e. feature) between 0 and 1
    def __init__(self, X=None, lower_bounds=None, upper_bounds=None):
        if lower_bounds is not None and upper_bounds is not None:
            self.min = np.array(lower_bounds)
            self.max = np.array(upper_bounds)
        elif X is not None:
            self.min = X.min(axis=0)
            self.max = X.max(axis=0)
        else:
            raise ValueError("Either X or both lower_bounds and upper_bounds must be provided.")

    def __call__(self, x):
        return (x - self.min) / (self.max - self.min + 1e-8)

    def inverse(self, x_normalized):
        return x_normalized * (self.max - self.min + 1e-8) + self.min

    def save(self, filepath):
      with open(filepath, 'wb') as f:
          pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

class MutationDataset(Dataset):
    # Create custom dataset
    def __init__(self, X, transform=None):
        self.X = X.astype(np.float32)
        self.y = np.zeros((len(self.X), 1), dtype=np.float32) # Dummy value since we do not use target
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]        
        if self.transform:
            x = self.transform(x)            
        return torch.from_numpy(x).float(), torch.from_numpy(self.y[idx]).float()

class TrainDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

# ----------------------
# MODEL
# ----------------------
class MutationNetwork(nn.Module):
    def __init__(self, input_dim):
        super(MutationNetwork, self).__init__()

        '''
        Receives a vector (individual) and returns a mutated vector (individual)
        Works with feature-wise normalized vectors (to [0,1])
        input_dim is the length of the vector (number of alleles of the individual)
        '''                
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),            
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.model(x)

# ----------------------
# TRAIN FUNCTIONS
# ----------------------
def prepare_data(input_data, train=True, test_size=0.2, batch_size=32, random_state=42, return_raw=False):
    # Prepare features
    data = input_data.copy()
    data = data[data['parent_fitness'] > data['mutation_fitness']] # Train only with good mutations
    X_raw = np.array(list(data['parent'].values))
    y_raw = np.array(list(data['mutation'].values))
    # Scenario configuration
    vector_size = len(X_raw[0])

    if train:
      # Split into training and evaluation sets
      X_train, X_eval, y_train, y_eval = train_test_split(X_raw, y_raw, test_size=test_size, random_state=random_state)

      # Fit the scaler on training set
      scaler = FeatureScaler(X_train)
      scaler.save('scaler.pickle') # Save fitted scaler for later use

      if return_raw:
        return X_train, y_train, X_eval, y_eval, vector_size

      # Create datasets and dataloaders
      train_dataset = TrainDataset(X_train, y_train, transform=scaler)
      print("Train set length:", len(train_dataset))
      train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

      eval_dataset = TrainDataset(X_eval, y_eval, transform=scaler)
      print("Validation set length:", len(eval_dataset))
      eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

      return train_dataloader, eval_dataloader, vector_size

    else:
      # Pass all the data as a single dataloader to perform inference
      scaler = FeatureScaler.load('scaler.pickle')
      dataset = MutationDataset(X_raw, y_raw, transform=scaler)
      print("Test set length:", len(dataset))
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

      return dataloader, vector_size

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    epochs=20,
    eval_every=5,
    lr=0.01,
    step_size=15,
    gamma=0.5,
    weight_decay=1e-4,
    display=False
):
    # Initialize hyperparameters
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_losses = []
    eval_losses = []

    for epoch in range(epochs):
        # Training loop
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Evaluate every `eval_every` epochs
        if (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs:
            model.eval()
            total_eval_loss = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                    total_eval_loss += loss.item()

            avg_eval_loss = total_eval_loss / len(val_loader)
            eval_losses.append((epoch + 1, avg_eval_loss))

            print(f"\n  -> Evaluation @ Epoch {epoch+1}: Eval Loss = {avg_eval_loss:.4f}\n")
            if (epoch + 1) == epochs:
              print("  -> Sample Predictions:")
              for x_batch, y_batch in val_loader:                  
                  outputs = model(x_batch)
                  loss = criterion(outputs, y_batch)
                  for original, pred_mut in zip(outputs[:3], y_batch[:3]):
                      print(f"     Original: {np.round(original.tolist(),3)}")
                      print(f"     Mutated: {np.round(pred_mut.tolist(),3)}\n")                      
                  break

    # Save the model state dictionary
    torch.save(model.state_dict(), 'mutator.pth')
    print("Model saved to mutator.pth")

    if display:
        # Plotting Loss Curves
        print()
        epochs_range = list(range(1, epochs + 1))
        eval_epochs, eval_values = zip(*eval_losses) if eval_losses else ([], [])

        plt.figure(figsize=(10, 5))
        plt.plot(epochs_range, train_losses, label="Train Loss")
        if eval_losses:
            plt.plot(eval_epochs, eval_values, 'ro-', label="Eval Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (logarithmic)")
        plt.yscale("log")
        plt.title("Training & Evaluation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ----------------------
# MAIN
# ----------------------
if __name__ == "__main__":
    # Load data
    raw_a = pd.read_parquet('training_data/1gpk_data.parquet')
    raw_b = pd.read_parquet('training_data/1gpk_data_2.parquet')
    raw_c = pd.read_parquet('training_data/1gpk_data_3.parquet')
    raw_d = pd.read_parquet('training_data/1gpk_data_4.parquet')
    raw_data = pd.concat([raw_a, raw_b, raw_c, raw_d])

    train, val, vector_size = prepare_data(raw_data)

    # Initialize model
    model = MutationNetwork(input_dim=vector_size)

    # Train model
    criterion = nn.MSELoss()

    train_model(model, train, val, criterion, epochs=5, eval_every=20, lr=0.03)