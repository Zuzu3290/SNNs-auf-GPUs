import torch
import torch.nn as nn
import numpy as np


class SNNTrainer:
    def __init__(self, model, train_loader, test_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = config
        self.device = device

        self.loss_fn = nn.CrossEntropyLoss()

        if self.cfg.OPTIMIZER.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.LEARNING_RATE,
                weight_decay=self.cfg.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg.OPTIMIZER}")

        self.train_loss_hist = []
        self.test_loss_hist = []
        self.train_acc_hist = []
        self.test_acc_hist = []

    def get_predictions(self, spk_rec):
        return spk_rec.sum(dim=0).argmax(dim=1)

    def evaluate(self, loader):
        self.model.eval()

        all_preds = []
        all_targets = []

        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for data, targets in loader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                spk_rec, mem_rec = self.model(data)

                loss = sum(
                    self.loss_fn(mem_rec[t], targets)
                    for t in range(self.cfg.TIMESTEPS)
                )

                total_loss += loss.item() / self.cfg.TIMESTEPS

                preds = self.get_predictions(spk_rec)

                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

                n_batches += 1

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        accuracy = (all_preds == all_targets).float().mean().item()
        avg_loss = total_loss / n_batches

        return all_preds, all_targets, avg_loss, accuracy

    def build_confusion_matrix(self, preds, targets):
        cm = np.zeros((self.cfg.NUM_CLASSES, self.cfg.NUM_CLASSES), dtype=int)

        for target, pred in zip(targets.numpy(), preds.numpy()):
            cm[target][pred] += 1

        return cm

    def compute_performance_matrix(self, cm):
        total = cm.sum()
        metrics = []

        for c in range(self.cfg.NUM_CLASSES):
            tp = int(cm[c, c])
            fp = int(cm[:, c].sum() - tp)
            fn = int(cm[c, :].sum() - tp)
            tn = int(total - tp - fp - fn)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            accuracy = (tp + tn) / total

            metrics.append({
                "class": c,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "specificity": round(specificity, 4),
                "accuracy": round(accuracy, 4)
            })

        return metrics

    def train(self):
        print("Pre-training evaluation started.")

        pre_preds, pre_targets, pre_loss, pre_acc = self.evaluate(self.test_loader)
        pre_cm = self.build_confusion_matrix(pre_preds, pre_targets)
        pre_metrics = self.compute_performance_matrix(pre_cm)

        print(f"Pre-training accuracy: {pre_acc * 100:.2f}%")

        print("Training started.")

        for epoch in range(self.cfg.EPOCHS):
            self.model.train()

            running_loss = 0.0
            running_acc = 0.0
            n_batches = 0

            for data, targets in self.train_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                spk_rec, mem_rec = self.model(data)

                loss = sum(
                    self.loss_fn(mem_rec[t], targets)
                    for t in range(self.cfg.TIMESTEPS)
                )

                loss.backward()
                self.optimizer.step()

                preds = self.get_predictions(spk_rec)

                running_loss += loss.item() / self.cfg.TIMESTEPS
                running_acc += (preds == targets).float().mean().item()
                n_batches += 1

            train_loss = running_loss / n_batches
            train_acc = running_acc / n_batches

            _, _, test_loss, test_acc = self.evaluate(self.test_loader)

            self.train_loss_hist.append(train_loss)
            self.train_acc_hist.append(train_acc)
            self.test_loss_hist.append(test_loss)
            self.test_acc_hist.append(test_acc)

            print(
                f"Epoch {epoch + 1}/{self.cfg.EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc * 100:.2f}% | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc * 100:.2f}%"
            )

        print("Post-training evaluation started.")

        post_preds, post_targets, post_loss, post_acc = self.evaluate(self.test_loader)
        post_cm = self.build_confusion_matrix(post_preds, post_targets)
        post_metrics = self.compute_performance_matrix(post_cm)

        print(f"Post-training accuracy: {post_acc * 100:.2f}%")

        return {
            "pre_loss": pre_loss,
            "pre_accuracy": pre_acc,
            "pre_confusion_matrix": pre_cm,
            "pre_metrics": pre_metrics,
            "post_loss": post_loss,
            "post_accuracy": post_acc,
            "post_confusion_matrix": post_cm,
            "post_metrics": post_metrics,
            "train_loss_history": self.train_loss_hist,
            "test_loss_history": self.test_loss_hist,
            "train_accuracy_history": self.train_acc_hist,
            "test_accuracy_history": self.test_acc_hist
        }