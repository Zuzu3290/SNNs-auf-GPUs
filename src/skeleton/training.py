import torch
import torch.nn as nn
import numpy as np
import os


class SNNTrainer:
    _REQUIRED_CFG = [
        "OPTIMIZER", "LEARNING_RATE", "WEIGHT_DECAY",
        "EPOCHS", "TIMESTEPS", "NUM_CLASSES"
    ]

    def __init__(self, model, train_loader, test_loader, config, device):
        # --- Validate config up front for clear error messages ---
        missing = [k for k in self._REQUIRED_CFG if not hasattr(config, k)]
        if missing:
            raise AttributeError(f"Config is missing required attributes: {missing}")

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = config
        self.device = device

        self.loss_fn = nn.CrossEntropyLoss()

        optimizer_name = self.cfg.OPTIMIZER.lower()
        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.LEARNING_RATE,
                weight_decay=self.cfg.WEIGHT_DECAY,
            )
        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.LEARNING_RATE,
                momentum=getattr(self.cfg, "MOMENTUM", 0.9),
                weight_decay=self.cfg.WEIGHT_DECAY,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg.OPTIMIZER}")

        # LR scheduler (optional — set SCHEDULER in config to enable)
        scheduler_name = getattr(self.cfg, "SCHEDULER", "none").lower()
        if scheduler_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.EPOCHS
            )
        elif scheduler_name == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=getattr(self.cfg, "SCHEDULER_STEP_SIZE", 10),
                gamma=getattr(self.cfg, "SCHEDULER_GAMMA", 0.5),
            )
        else:
            self.scheduler = None

        # Gradient clipping (optional — set GRAD_CLIP_NORM in config to enable)
        self.grad_clip_norm = getattr(self.cfg, "GRAD_CLIP_NORM", None)

        self.train_loss_hist = []
        self.test_loss_hist = []
        self.train_acc_hist = []
        self.test_acc_hist = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_predictions(self, spk_rec):
        return spk_rec.sum(dim=0).argmax(dim=1)

    def _save_checkpoint(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss_hist": self.train_loss_hist,
                "test_loss_hist": self.test_loss_hist,
                "train_acc_hist": self.train_acc_hist,
                "test_acc_hist": self.test_acc_hist,
            },
            path,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

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

                # FIX: divide by TIMESTEPS here so the scale matches
                # what is accumulated during training
                loss = (
                    sum(self.loss_fn(mem_rec[t], targets) for t in range(self.cfg.TIMESTEPS))
                    / self.cfg.TIMESTEPS
                )

                total_loss += loss.item()

                preds = self.get_predictions(spk_rec)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

                n_batches += 1

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        accuracy = (all_preds == all_targets).float().mean().item()
        avg_loss = total_loss / n_batches

        return all_preds, all_targets, avg_loss, accuracy

    # ------------------------------------------------------------------
    # Confusion matrix & metrics
    # ------------------------------------------------------------------

    def build_confusion_matrix(self, preds, targets):
        """Vectorised confusion matrix — no Python loop over samples."""
        cm = np.zeros((self.cfg.NUM_CLASSES, self.cfg.NUM_CLASSES), dtype=int)
        np.add.at(cm, (targets.numpy(), preds.numpy()), 1)
        return cm

    def compute_performance_matrix(self, cm):
        total = cm.sum()          # computed once, outside the loop
        metrics = []

        for c in range(self.cfg.NUM_CLASSES):
            tp = int(cm[c, c])
            fp = int(cm[:, c].sum() - tp)
            fn = int(cm[c, :].sum() - tp)
            tn = int(total - tp - fp - fn)

            precision   = tp / (tp + fp)   if (tp + fp) > 0           else 0.0
            recall      = tp / (tp + fn)   if (tp + fn) > 0           else 0.0
            f1          = (2 * precision * recall / (precision + recall)
                           if (precision + recall) > 0 else 0.0)
            specificity = tn / (tn + fp)   if (tn + fp) > 0           else 0.0
            accuracy    = (tp + tn) / total

            metrics.append({
                "class":       c,
                "TP":          tp,
                "FP":          fp,
                "FN":          fn,
                "TN":          tn,
                "precision":   round(precision,   4),
                "recall":      round(recall,       4),
                "f1":          round(f1,           4),
                "specificity": round(specificity,  4),
                "accuracy":    round(accuracy,     4),
            })

        return metrics

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, checkpoint_dir=None, checkpoint_name="best_model.pt"):
        """
        Train the SNN and return pre/post metrics plus history.

        Args:
            checkpoint_dir  : directory where the best checkpoint is saved.
                              Pass None to skip checkpointing.
            checkpoint_name : filename for the checkpoint file.
        """
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        else:
            checkpoint_path = None

        # ---- Pre-training evaluation ----
        print("Pre-training evaluation started.")
        pre_preds, pre_targets, pre_loss, pre_acc = self.evaluate(self.test_loader)
        pre_cm      = self.build_confusion_matrix(pre_preds, pre_targets)
        pre_metrics = self.compute_performance_matrix(pre_cm)
        print(f"Pre-training accuracy: {pre_acc * 100:.2f}%")

        # ---- Training loop ----
        print("Training started.")
        best_test_acc = 0.0

        for epoch in range(self.cfg.EPOCHS):
            self.model.train()

            running_loss = 0.0
            running_acc  = 0.0
            n_batches    = 0

            for data, targets in self.train_loader:
                data    = data.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                spk_rec, mem_rec = self.model(data)

                # FIX: divide by TIMESTEPS so train/test loss is on the same scale
                loss = (
                    sum(self.loss_fn(mem_rec[t], targets) for t in range(self.cfg.TIMESTEPS))
                    / self.cfg.TIMESTEPS
                )

                loss.backward()

                # Optional gradient clipping
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )

                self.optimizer.step()

                preds = self.get_predictions(spk_rec)
                running_loss += loss.item()
                running_acc  += (preds == targets).float().mean().item()
                n_batches    += 1

            train_loss = running_loss / n_batches
            train_acc  = running_acc  / n_batches

            _, _, test_loss, test_acc = self.evaluate(self.test_loader)

            self.train_loss_hist.append(train_loss)
            self.train_acc_hist.append(train_acc)
            self.test_loss_hist.append(test_loss)
            self.test_acc_hist.append(test_acc)

            # Step LR scheduler if one is configured
            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Epoch {epoch + 1}/{self.cfg.EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc * 100:.2f}% | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc * 100:.2f}%"
            )

            # Save best checkpoint
            if checkpoint_path is not None and test_acc > best_test_acc:
                best_test_acc = test_acc
                self._save_checkpoint(checkpoint_path)
                print(f"  → New best ({best_test_acc * 100:.2f}%), checkpoint saved.")

        # ---- Post-training evaluation ----
        print("Post-training evaluation started.")
        post_preds, post_targets, post_loss, post_acc = self.evaluate(self.test_loader)
        post_cm      = self.build_confusion_matrix(post_preds, post_targets)
        post_metrics = self.compute_performance_matrix(post_cm)
        print(f"Post-training accuracy: {post_acc * 100:.2f}%")

        return {
            "pre_loss":              pre_loss,
            "pre_accuracy":          pre_acc,
            "pre_confusion_matrix":  pre_cm,
            "pre_metrics":           pre_metrics,
            "post_loss":             post_loss,
            "post_accuracy":         post_acc,
            "post_confusion_matrix": post_cm,
            "post_metrics":          post_metrics,
            "train_loss_history":    self.train_loss_hist,
            "test_loss_history":     self.test_loss_hist,
            "train_accuracy_history": self.train_acc_hist,
            "test_accuracy_history": self.test_acc_hist,
        }