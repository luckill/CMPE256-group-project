import torch
import torch.nn as nn

class neural_cllaborative_filtering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dimension = 32, use_sigmoid=False):
        super().__init__()
        self.use_sigmoid = use_sigmoid

        self.users = nn.Embedding(num_users, embedding_dimension)
        self.items = nn.Embedding(num_items, embedding_dimension)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dimension * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

        if self.use_sigmoid:
            self.output_activation = nn.Sigmoid()

    def forward(self, user_id, item_id):
        user_embedding = self.users(user_id)
        item_embedding = self.items(item_id)
        concatenated = torch.cat([user_embedding, item_embedding], dim=-1)
        output = self.mlp(concatenated).squeeze(-1)
        if self.use_sigmoid:
            output = self.output_activation(output)
        return output
