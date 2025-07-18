import torch
from llm_config import cfg
from sklearn.linear_model import LogisticRegression
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class LayerClassifier:
    def __init__(self, llm_cfg: cfg, lr: float = 0.01, max_iter: int = 10000):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[DEBUG] Layer Classifier Using device: {self.device}")
        self.linear = LogisticRegression(solver="saga", max_iter=max_iter)

        self.data = {
            "train": {
                "pos": None,
                "neg": None,
            },
            "test": {
                "pos": None,
                "neg": None,
            },
        }

    def train(
        self,
        pos_tensor: torch.tensor,
        neg_tensor: torch.tensor,
        n_epoch: int = 100,
        batch_size: int = 64,
    ) -> list[float]:
        X = torch.vstack([pos_tensor, neg_tensor]).to(self.device)
        y = torch.cat(
            (torch.ones(pos_tensor.size(0)), torch.zeros(neg_tensor.size(0)))
        ).to(self.device)

        self.data["train"]["pos"] = pos_tensor.cpu()
        self.data["train"]["neg"] = neg_tensor.cpu()

        self.linear.fit(X.cpu().numpy(), y.cpu().numpy())

        return []

    def predict(self, tensor: torch.tensor) -> torch.tensor:
        return torch.tensor(self.linear.predict(tensor.cpu().numpy()))

    def predict_proba(self, tensor: torch.tensor) -> torch.tensor:
        w, b = self.get_weights_bias()
        return torch.sigmoid(tensor.to(self.device).to(w.dtype) @ w.T + b)
    
    def distance_to_hyperplane(self, tensor: torch.Tensor) -> torch.Tensor:
        w, b = self.get_weights_bias()
        logit = tensor.to(self.device).to(w.dtype) @ w.T + b
        norm = torch.norm(w)
        return logit / norm

    def evaluate_testacc(
        self, pos_tensor: torch.tensor=None, neg_tensor: torch.tensor=None
    ) -> float:
        if pos_tensor is None or neg_tensor is None:
            assert (self.data["test"]["pos"] is not None) and (
                self.data["test"]["neg"] is not None
            ), "No test data provided. Please provide pos_tensor and neg_tensor or use the stored test data."
            pos_tensor = self.data["test"]["pos"]
            neg_tensor = self.data["test"]["neg"]
            
        test_data = torch.vstack([pos_tensor, neg_tensor]).to(self.device)
        predictions = self.predict(test_data)
        true_labels = torch.cat(
            (torch.ones(pos_tensor.size(0)), torch.zeros(neg_tensor.size(0)))
        )

        
        # correct_count = torch.sum((predictions > 0.5) == true_labels).item()

        # self.data["test"]["pos"] = pos_tensor.cpu()
        # self.data["test"]["neg"] = neg_tensor.cpu()

        # return correct_count / len(true_labels)

        # Convert predictions to binary labels
        pred_labels = (predictions > 0.5).int()

        # Compute metrics
        acc = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)

        # Save test data
        self.data["test"]["pos"] = pos_tensor.cpu()
        self.data["test"]["neg"] = neg_tensor.cpu()

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def get_weights_bias(self) -> tuple[torch.tensor, torch.tensor]:
        return torch.tensor(self.linear.coef_).to(self.device), torch.tensor(
            self.linear.intercept_
        ).to(self.device)
