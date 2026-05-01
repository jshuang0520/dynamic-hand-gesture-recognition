import torch
from utilities.logger import get_logger

log = get_logger("TRAINER")

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, quick_test=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.quick_test = quick_test
        if self.quick_test:
            log("QUICK_TEST is ENABLED. Truncating loops to 2 batches.", "WARNING")

    def run(self, mode="gpu", epochs=10):
        device = torch.device("cuda:0" if mode == "gpu" and torch.cuda.is_available() else "cpu")
        log(f"Hardware allocated: {device}")
        
        self.model.to(device)
        actual_epochs = 1 if self.quick_test else epochs

        for epoch in range(1, actual_epochs + 1):
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                if self.quick_test and batch_idx >= 2: break
                
                data, targets = data.to(device), targets.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
            log(f"[EPOCH {epoch}/{actual_epochs}] Train Loss: {train_loss / (batch_idx + 1):.4f}")
            
        log("Training phase complete.")
        return self.model