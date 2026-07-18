import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import yaml 
# transformer block we created
from src.model.transformer_block import TransformerBlock
from src.model.dataset import FinancialDataset 
from src.model.trainer import FinancialTrainer

#1. pytorch profiling step 1 -> Pytorch Profiler for performance measurement(record for future hardware acceleration part)
from torch.profiler import profile , record_function, ProfilerActivity, schedule

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


def train_financial_engine():

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
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
    

    #2. pytorch profiling step 2 -> Define profiler schedule
    profiler_schedule = schedule (
        skip_first = 3,
        wait = 1,
        warmup = 1,
        active = 2,
        repeat = 1
    )
    print(f"\n Start profiling ➔ device: {device.type.upper()}")
    # training
    trainer = FinancialTrainer(model, criterion, optimizer, device, scaler)

    #3. pytorch profiling step3 -> analyze calculation from cpu, schedule, memory and tensor shapes
    with profile (
        activities = [ProfilerActivity.CPU],#currently using CPU since using github codespace 
        schedule = profiler_schedule,
        profile_memory=False,
        record_shapes=False
    ) as prof:
#Profiling_epoch
        trainer.profiling_epoch(train_loader, prof)

    #4. pytorch profiling step4 -> hardware report file
    output_path = "docs/profiling_raw_cpu.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        report_table = prof.key_averages().table(
            sort_by="self_cpu_time_total", 
            row_limit=10  
        )
        f.write(report_table)
        f.flush() 
    print("finished profiling!!")

    

#Train_epoch 
    # epochs = 5
    # for epoch in range(epochs):
    #     avg_loss, epoch_time = trainer.train_epoch(train_loader, epoch, epochs)
    #     print(f"Epoch [{epoch+1}/{epochs}] finished! average Loss: {avg_loss:.4f} | total time: {epoch_time:.2f}seconds")
    #     print("-"*60)

    # print("\nfinished training!")
    # print("="*60)

if __name__ == "__main__":
    train_financial_engine()