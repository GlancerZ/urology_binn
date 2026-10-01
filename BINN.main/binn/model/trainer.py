import torch
import torch.nn.functional as F


class BINNTrainer:

    def __init__(self, binn_model, save_dir: str = "", class_weights=None):
        self.save_dir      = save_dir
        self.network       = binn_model
        self.logger        = BINNLogger(save_dir=save_dir)
        self.class_weights = class_weights   # ← 修正1

    def fit(self, dataloaders, num_epochs=30, learning_rate=1e-4,
            checkpoint_path=None):

        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        weight = self.class_weights.to(self.network.device) \
                 if self.class_weights is not None else None

        for epoch in range(num_epochs):
            self.network.train()
            train_loss, train_accuracy = 0.0, 0.0

            for inputs, targets in dataloaders["train"]:
                inputs  = inputs.to(self.network.device)
                targets = targets.to(self.network.device)

                optimizer.zero_grad()
                outputs = self.network(inputs)
                loss = F.cross_entropy(outputs, targets, weight=weight)
                loss.backward()
                optimizer.step()

                train_loss     += loss.item()
                train_accuracy += (torch.argmax(outputs, dim=1) == targets).float().mean().item()

            avg_train_loss     = train_loss / len(dataloaders["train"])
            avg_train_accuracy = train_accuracy / len(dataloaders["train"])
            print(f"[Epoch {epoch+1}/{num_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Train Accuracy: {avg_train_accuracy:.4f}")

            if "val" in dataloaders:
                self.network.eval()
                val_loss, val_accuracy = 0.0, 0.0
                with torch.no_grad():
                    for inputs, targets in dataloaders["val"]:
                        inputs  = inputs.to(self.network.device)
                        targets = targets.to(self.network.device)
                        outputs = self.network(inputs)
                        loss = F.cross_entropy(outputs, targets, weight=weight)  # ← 修正2
                        val_loss     += loss.item()
                        val_accuracy += (torch.argmax(outputs, dim=1) == targets).float().mean().item()

                avg_val_loss     = val_loss / len(dataloaders["val"])
                avg_val_accuracy = val_accuracy / len(dataloaders["val"])
                print(f"[Epoch {epoch+1}/{num_epochs}] "
                      f"Val Loss: {avg_val_loss:.4f}, "
                      f"Val Accuracy: {avg_val_accuracy:.4f}")

            if checkpoint_path:
                torch.save(self.network.state_dict(),
                           f"{checkpoint_path}_epoch{epoch+1}.pt")

    def evaluate(self, dataloader):
        self.network.eval()
        total_loss, total_accuracy = 0.0, 0.0
        weight = self.class_weights.to(self.network.device) \
                 if self.class_weights is not None else None

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs  = inputs.to(self.network.device)
                targets = targets.to(self.network.device)
                outputs = self.network(inputs)
                loss = F.cross_entropy(outputs, targets, weight=weight)  # ← 修正3
                total_loss     += loss.item()
                total_accuracy += (torch.argmax(outputs, dim=1) == targets).float().mean().item()

        avg_loss     = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)
        return {"loss": avg_loss, "accuracy": avg_accuracy}

    def update_model(self, new_binn_model):
        self.binn_model = new_binn_model
        self.logger     = BINNLogger(save_dir=self.save_dir)


class BINNLogger:

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.logs = {"train": [], "val": []}

    def log(self, phase, metrics):
        self.logs[phase].append(metrics)

    def save_logs(self):
        import pandas as pd
        for phase, log_data in self.logs.items():
            if log_data:
                df = pd.DataFrame(log_data)
                df.to_csv(f"{self.save_dir}/{phase}_logs.csv", index=False)