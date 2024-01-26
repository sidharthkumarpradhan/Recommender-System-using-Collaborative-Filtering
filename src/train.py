# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collaborative_filtering_model import CollaborativeFilteringModel

# Load the preprocessed data
train_df = pd.read_csv('data/train_ratings.csv')

# Initialize model, optimizer, and criterion
model = CollaborativeFilteringModel(num_users, num_items, embedding_dim=50)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

# Move model to device
model = model.to(device)
criterion = criterion.to(device)

# Create PyTorch DataLoader for training
train_dataset = RatingDataset(train_df, user_dict, item_dict)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training loop
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()
        user_ids, item_ids, ratings = batch
        predictions = model(user_ids, item_ids)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

# Number of epochs for training
N_EPOCHS = 10

# Train the model
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    print(f'Epoch: {epoch + 1}, Training Loss: {train_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'saved_model.pt')
