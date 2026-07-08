import torch
import torch.nn as nn
import math
class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        # Create num_heads SingleHeadAttention instances using nn.ModuleList
        # Each head size = attention_dim // num_heads
        # Use: self.SingleHeadAttention(embedding_dim, head_size)
        # After the heads, add an output projection: nn.Linear(attention_dim, attention_dim, bias=False)
        head_size = attention_dim // num_heads
        self.heads= nn.ModuleList([
            self.SingleHeadAttention(embedding_dim, head_size) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(attention_dim, attention_dim, bias=False)
        pass

    def forward(self, embedded):
        # Run each head on the input, concatenate outputs along dim=2
        # Pass concatenated result through the output projection (W_O)
        # Return result rounded to 4 decimal places
        output = []
        #output = [[B, T ,head_size], [B, T, head_size] .......]
        for head in self.heads:
            #[B, T, head_size]
            output.append(head(embedded))
        
        #after cat -> [B, T ,A] torch.cat사용시 저절로 tensor차원으로 변경됨
        concatenated_res = torch.cat(output, dim=2)
        return torch.round(self.proj(concatenated_res), decimals=4)

    #[B,T,A]
    class SingleHeadAttention(nn.Module):
        #__init__ 학습을 통해 값이 계속 변하고 유지되어야 하는 레이어. 모델 전체에서 고정되어 쓰이는 부분 (ex_ weight matrix)
        #Embedding_dim = E, attention_dim = A
        def __init__(self, embedding_dim, attention_dim):
            super().__init__()
            torch.manual_seed(0)
            #to use it as a gloabal variable in forward function
            self.attention_dim = attention_dim
            # Create three linear projections (Key, Query, Value) with bias=False
            # Instantiation order matters for reproducible weights: key, query, value

            #weight matrix [embedding_dim , attention_dim]
            self.proj_k = nn.Linear(embedding_dim, attention_dim, bias = False)
            self.proj_q = nn.Linear(embedding_dim, attention_dim, bias = False)
            self.proj_v = nn.Linear(embedding_dim, attention_dim, bias = False)
            
        #forward: 입력 데이터 들어왔을때만 실시간으로 계산되고 사라지는 단순 수학연산이나 함수
        # embedded = [B, T, E]
        def forward(self, embedded):
            
            # 1. Project input through K, Q, V linear layers
            # [B , T, A]
            queries = self.proj_q(embedded) 
            keys = self.proj_k(embedded)
            values = self.proj_v(embedded)

        
            # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim) -> Using permute or transpose func
            # attention_score = queries @ torch.permute(keys, (0,2,1)) / math.sqrt(attention_dim)

            # [B, T, T]
            attention_score = (queries @ keys.transpose(-2, -1) ) / math.sqrt(self.attention_dim)
            # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
            #    then masked_fill positions where mask == 0 with float('-inf')
            seq_len = embedded.size(1)

            # [T, T]
            causal_mask = torch.tril(torch.ones((seq_len, seq_len)))
            #used broadcasting 
            # [B, T, T]
            masked_score = attention_score.masked_fill(causal_mask==0, float('-inf'))
            # 4. Apply softmax(dim=2) to masked scores
            # [B, T ,T]
            attention_weights = torch.softmax(masked_score, dim=-1)
        
            
            # 5. Return (scores @ V) rounded to 4 decimal places
            #[B, T, T] @ [B, T, A] -> res = [B, T ,A]
            return torch.round(attention_weights @ values , decimals=4)



