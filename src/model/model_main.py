import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import yaml 
# transformer block we created
from src.model.transformer_block import TransformerBlock
from src.model.dataset import FinancialDataset 

class UltimateFinancialTransformer(nn.Module):
    """
    A final end-to-end architecture based on Transformer Blocks, designed to process 
    8-dimensional financial data stacked in 24 layers to predict buy, hold, or sell signals
    """
    def __init__(self, input_dim=8, model_dim=512, num_heads=8, num_layers=24, num_classes=3):
        super().__init__()
        # (8 -> 512)
        self.input_projection = nn.Linear(input_dim, model_dim)
        
        # 24 layers
        self.layers = nn.ModuleList([
            TransformerBlock(model_dim=model_dim, num_heads=num_heads) for _ in range(num_layers)
        ])
        
        # norm layer and final classifier 
        self.final_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, num_classes)
        
    def forward(self, x):
        # x : [Batch_Size, Seq_Len(20), Features(8)]
        x = self.input_projection(x) # -> [Batch_Size, 20, 512]
        
        #24 layers
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_norm(x)
        
        # only print current timestamp(last)
        final_timestep = x[:, -1, :] # -> [Batch_Size, 512]
        return self.classifier(final_timestep) # -> [Batch_Size, 3]


def train_real_financial_engine():

    torch.manual_seed(0)
    print("="*60)
    print("Injecting financial data and launching 24-Layer Transformer")
    print("="*60)
    
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)["data_pipeline"]
    
   
    db_path = cfg["db_path"]
    ticker = cfg["tickers"][0] 
    window_size = 20           
    
    print(f" finished yaml loading -> DB: {db_path} | Ticker: {ticker}")
    print("loading window and label from sqlite database")
    
    # loading state
    train_dataset = FinancialDataset(db_path=db_path, ticker=ticker, window_size=window_size) 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    
    # hardware setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] current device: {device}")
    print(f"[INFO] total training data batch: {len(train_loader)}개")
    
    #model and optimization instances
    model = UltimateFinancialTransformer(num_layers=24).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Mixed Precision (FP16) scaler
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # training
    model.train()
    epochs = 5
    
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # inputs: financial data tensors, targets: buy/sell label data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # FP16 혼합 정밀도로 텐서 코어 가속 가동
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) # loss calculation
                scaler.scale(loss).backward() # backpropagation
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"   📊 Epoch [{epoch+1}/{epochs}] | Step [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")
                
        epoch_time = time.time() - start_time
        print(f"🔥 Epoch [{epoch+1}/{epochs}] finished! average Loss: {running_loss/len(train_loader):.4f} | total time: {epoch_time:.2f}seconds")
        print("-"*60)

    print("\nfinished training!")
    print("="*60)

if __name__ == "__main__":
    train_real_financial_engine()