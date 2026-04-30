import torch


class SNNInference:
    def __init__(self, model, config, device):
        self.model = model
        self.cfg = config
        self.device = device

    def predict_batch(self, data):
        self.model.eval()

        data = data.to(self.device)

        with torch.no_grad():
            spk_rec, mem_rec = self.model(data)

        predictions = spk_rec.sum(dim=0).argmax(dim=1)

        return predictions, spk_rec, mem_rec

    def predict_single(self, sample):
        sample = sample.unsqueeze(0).to(self.device)

        with torch.no_grad():
            spk_rec, mem_rec = self.model(sample)

        prediction = spk_rec.sum(dim=0).argmax(dim=1)

        return prediction.item(), spk_rec, mem_rec

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()