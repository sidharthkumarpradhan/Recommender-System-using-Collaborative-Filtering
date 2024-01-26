# collaborative_filtering_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CollaborativeFilteringModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50):
        super(CollaborativeFilteringModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        dot_product = torch.sum(user_embedding * item_embedding, dim=1)
        return dot_product
