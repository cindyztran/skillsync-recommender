import torch;
import torch.nn as nn;
import pandas as pd;
from sklearn.model_selection import train_test_split;
# import os

# simulated dataset
data = {
  "UserID": [1, 1, 2, 2, 3],
  "SkillID": [101, 102, 101, 103, 102],
  "Rating": [4, 5, 3, 4, 2],
}

df = pd.DataFrame(data);

# encode UserID and SkillID to integers
user_mapping = { id: idx for idx, id in enumerate(df["UserID"].unique()) }
skill_mapping = { id: idx for idx, id in enumerate(df["SkillID"].unique()) }

df["UserID"] = df["UserID"].map(user_mapping);
df["SkillID"] = df["SkillID"].map(skill_mapping);

# split data into train and test
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# pytorch dataset
class RecommendationDataset(torch.utils.data.Dataset):
  def __init__(self, data):
    self.users = torch.tensor(data["UserID"].values, dtype=torch.long);
    self.skills = torch.tensor(data["SkillID"].values, dtype=torch.long);
    self.ratings = torch.tensor(data["Rating"].values, dtype=torch.float32);

  def __len__(self):
    return len(self.users);

  def __getitem__(self, idx):
    return self.users[idx], self.skills[idx], self.ratings[idx];

train_dataset = RecommendationDataset(train_data);
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True);

# model definition
class RecommendationModel(nn.Module):
  def __init__(self, num_users, num_skills, embedding_dim=8):
    super(RecommendationModel, self).__init__();
    self.user_embedding = nn.Embedding(num_users, embedding_dim);
    self.skill_embedding = nn.Embedding(num_skills, embedding_dim);
    self.fc = nn.Linear(embedding_dim * 2, 1);

  def forward(self, user, skill):
    user_embed = self.user_embedding(user);
    skill_embed = self.skill_embedding(skill);
    combined = torch.cat([user_embed, skill_embed], dim=1);
    return self.fc(combined).squeeze();

# initialize model, loss and optimizer
num_users = len(user_mapping);
num_skills = len(skill_mapping);
model = RecommendationModel(num_users, num_skills);
criterion = nn.MSELoss();
optimizer = torch.optim.Adam(model.parameters(), lr=0.01);

# training loop
epochs = 10
for epoch in range(epochs):
  for user, skill, rating in train_loader:
    optimizer.zero_grad();
    predictions = model(user, skill);
    loss = criterion(predictions, rating);
    loss.backward();
    optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}");

# # Define the path to the 'models' directory
# model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))

# # Ensure the directory exists
# os.makedirs(model_dir, exist_ok=True)

# # Save the model
model_path = "models/recommendation_model.pth"
torch.save(model.state_dict(), model_path);
print(f"Model saved successfully to {model_path}")
