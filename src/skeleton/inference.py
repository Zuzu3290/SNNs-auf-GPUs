import torch


class SNNInference:
    _REQUIRED_CFG = ["TIMESTEPS", "NUM_CLASSES"]

    def __init__(self, model, config, device):
        missing = [k for k in self._REQUIRED_CFG if not hasattr(config, k)]
        if missing:
            raise AttributeError(f"Config is missing required attributes: {missing}")

        self.model = model
        self.cfg = config
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_predictions_and_confidence(self, spk_rec):
        """
        Returns class predictions and a normalised confidence score.

        Confidence is the fraction of total spikes attributed to the
        winning class — a cheap, interpretable proxy for certainty.

        Args:
            spk_rec : Tensor [T, B, C]  — spike recordings

        Returns:
            predictions : LongTensor  [B]
            confidence  : FloatTensor [B]   values in (0, 1]
        """
        spike_counts = spk_rec.sum(dim=0)           # [B, C]
        predictions  = spike_counts.argmax(dim=1)   # [B]

        total_spikes = spike_counts.sum(dim=1).clamp(min=1)           # [B]
        winning_spikes = spike_counts[range(len(predictions)), predictions]  # [B]
        confidence = winning_spikes / total_spikes                     # [B]

        return predictions, confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_batch(self, data):
        """
        Run inference on a batch of samples.

        Args:
            data : Tensor [B, ...]

        Returns:
            predictions : LongTensor  [B]
            confidence  : FloatTensor [B]
            spk_rec     : Tensor [T, B, C]
            mem_rec     : Tensor [T, B, C]
        """
        self.model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            spk_rec, mem_rec = self.model(data)

        predictions, confidence = self._get_predictions_and_confidence(spk_rec)

        return predictions, confidence, spk_rec, mem_rec

    def predict_single(self, sample):
        """
        Run inference on a single unbatched sample.

        Args:
            sample : Tensor [...]   — no batch dimension

        Returns:
            prediction : int
            confidence : float
            spk_rec    : Tensor [T, 1, C]
            mem_rec    : Tensor [T, 1, C]
        """
        self.model.eval()
        sample = sample.unsqueeze(0).to(self.device)

        with torch.no_grad():
            spk_rec, mem_rec = self.model(sample)

        predictions, confidence = self._get_predictions_and_confidence(spk_rec)

        return predictions[0].item(), confidence[0].item(), spk_rec, mem_rec

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path, full_checkpoint=False):
        """
        Save the model to disk.

        Args:
            path            : file path to save to
            full_checkpoint : if True, saves weights + config metadata
                              alongside the state dict, for traceability
        """
        if full_checkpoint:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "num_classes":      self.cfg.NUM_CLASSES,
                    "timesteps":        self.cfg.TIMESTEPS,
                },
                path,
            )
        else:
            torch.save(self.model.state_dict(), path)

    def load_model(self, path, full_checkpoint=False):
        """
        Load model weights from disk.

        Args:
            path            : file path to load from
            full_checkpoint : set True if the file was saved with
                              full_checkpoint=True in save_model()
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        state_dict = checkpoint["model_state_dict"] if full_checkpoint else checkpoint
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()