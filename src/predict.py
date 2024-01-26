# predict.py
import torch
from torch.utils.data import DataLoader
from collaborative_filtering_model import CollaborativeFilteringModel

# Load the preprocessed data
test_df = pd.read_csv('data/test_ratings.csv')

# Initialize model
model = CollaborativeFilteringModel(num_users, num_items, embedding_dim=50)
model.load_state_dict(torch.load('saved_model.pt'))
model.eval()

# Create PyTorch DataLoader for testing
test_dataset = RatingDataset(test_df, user_dict, item_dict)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Inference function
def predict_ratings(model, test_loader):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            user_ids, item_ids, _ = batch
            predictions = model(user_ids, item_ids)
            all_predictions.extend(predictions.cpu().numpy())

    return all_predictions

# Make predictions on the test set
test_predictions = predict_ratings(model, test_loader)
print(test_predictions)
