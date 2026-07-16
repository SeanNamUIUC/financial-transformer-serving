import torch
import torch.nn as nn

from src.model.attention import MultiHeadedSelfAttention

# Uses Pre-LN architecture: LayerNorm is applied BEFORE each sub-layer, not after.
# This differs from the original "Attention is All You Need" diagram but trains better.
class TransformerBlock(nn.Module):

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        # Instantiate in this order:
        # 1. self.MultiHeadedSelfAttention(model_dim, num_heads)
        self.multiheaded_attention = MultiHeadedSelfAttention(embedding_dim=model_dim, attention_dim=model_dim,num_heads= num_heads)
        # 2. self.VanillaNeuralNetwork(model_dim)
        self.vanilla_nn = self.VanillaNeuralNetwork(model_dim)
        # 3. Two nn.LayerNorm(model_dim) instances
        self.layer_norm = nn.LayerNorm(model_dim)
        

    def forward(self, embedded):
        # Two residual connections with Pre-LN:
        #   x = x + attention(layer_norm_1(x))
        embedded = embedded + self.multiheaded_attention(self.layer_norm(embedded))
        #   x = x + feed_forward(layer_norm_2(x))
        embedded = embedded + self.vanilla_nn(self.layer_norm(embedded))
        # Return result rounded to 4 decimal places
        return torch.round(embedded, decimals=4)
        


    class VanillaNeuralNetwork(nn.Module):

        def __init__(self, model_dim: int):
            super().__init__()
            self.up_projection = nn.Linear(model_dim, model_dim * 4)
            self.relu = nn.ReLU()
            self.down_projection = nn.Linear(model_dim * 4, model_dim)
            self.dropout = nn.Dropout(0.2) # using p = 0.2

        def forward(self, x):
            return self.dropout(self.down_projection(self.relu(self.up_projection(x))))
