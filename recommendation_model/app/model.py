import torch
import torch.nn as nn

# example data
user_mapping = {1: 0, 2: 1, 3: 2}
skill_mapping = {101: 0, 102: 1, 103: 2}

# model definition
class RecommendationModel(nn.Module):
    def __init__(self, num_users, num_skills, embedding_dim=8):
        super(RecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.skill_embedding = nn.Embedding(num_skills, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user, skill):
        user_embed = self.user_embedding(user)
        skill_embed = self.skill_embedding(skill)
        combined = torch.cat([user_embed, skill_embed], dim=-1)
        return self.fc(combined).squeeze()