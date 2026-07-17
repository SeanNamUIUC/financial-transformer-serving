import torch
import time

class FinancialTrainer:
    def __init__(self, model, criterion, optimizer, device, scaler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer 
        self.device = device
        self.scaler = scaler

    def train_epoch(self, train_loader, epoch, epochs):
        #1epoch 동안 훈련담당
        self.model.train()#모델을 훈련모드로. nn.module 내의 training flag = True -> activate dropout function
        running_loss = 0.0
        start_time = time.time()
        # mini batch training 
        for batch_idx , (inputs, targets) in enumerate (train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()# 가중치 텐서 (weight tensor 내부의 weight.grad값들을 0으로 초기화)

            if self.scaler:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            running_loss += loss.item()
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")
        epoch_time = time.time() - start_time
        avg_loss = running_loss / len(train_loader)# running_loss / 총 배치 개수
        return avg_loss, epoch_time
            
