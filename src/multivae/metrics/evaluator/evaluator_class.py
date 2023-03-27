import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluator_config import CoherenceEvaluatorConfig


class CoherenceEvaluator:
    def __init__(
        self,
        model,
        classifiers,
        test_dataset,
        output=None,
        eval_config=CoherenceEvaluatorConfig(),
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(self.device)
        self.clfs = classifiers
        self.n_data = len(test_dataset)
        self.test_loader = DataLoader(test_dataset, batch_size=eval_config.batch_size)
        if output is not None:
            self.f = open(output + "/metrics.txt", "w+")
            print("Writing results in ", self.f)

        for k in self.clfs:
            self.clfs[k] = self.clfs[k].to(self.device)

    def pair_accuracies(self):
        accuracies = {}
        for batch in self.test_loader:
            batch.data = {m: batch.data[m].to(self.device) for m in batch.data}
            batch.labels = batch.labels.to(self.device)
            for cond_m in self.model.encoders:
                for pred_m in self.model.encoders:
                    output = self.model.predict(batch, cond_m, pred_m)
                    preds = self.clfs[pred_m](output[pred_m])
                    pred_labels = torch.argmax(preds, dim=1)
                    try:
                        accuracies[cond_m + "_" + pred_m] += torch.sum(
                            pred_labels == batch.labels
                        )
                    except:
                        accuracies[cond_m + "_" + pred_m] = torch.sum(
                            pred_labels == batch.labels
                        )

        acc = {k: accuracies[k].cpu().numpy() / self.n_data for k in accuracies}
        if hasattr(self, "f"):
            self.f.write("Pair accuracies \n")
            mean_pair_acc = np.mean(list(acc.values()))
            self.f.write(acc.__str__() + "\n")
            self.f.write("Mean pair accuracies" + str(mean_pair_acc))
        return acc

    def all_one_accuracies(self):
        accuracies = {}
        for batch in self.test_loader:
            batch.data = {m: batch.data[m].to(self.device) for m in batch.data}
            batch.labels = batch.labels.to(self.device)
            for pred_m in self.model.encoders:
                cond_m = [m for m in self.model.encoders if m != pred_m]
                output = self.model.predict(batch, cond_m, pred_m)
                preds = self.clfs[pred_m](output[pred_m])
                pred_labels = torch.argmax(preds, dim=1)
                try:
                    accuracies[pred_m] += torch.sum(pred_labels == batch.labels)
                except:
                    accuracies[pred_m] = torch.sum(pred_labels == batch.labels)

        acc = {k: accuracies[k].cpu().numpy() / self.n_data for k in accuracies}
        if hasattr(self, "f"):
            self.f.write("All to one accuracies \n")
            mean_pair_acc = np.mean(list(acc.values()))
            self.f.write(acc.__str__() + "\n")
            self.f.write("Mean all-to-one accuracies" + str(mean_pair_acc))
        return acc

    def joint_nll(self):
        ll = 0
        nb_batch = 0
        for batch in self.test_loader:
            batch.data = {m: batch.data[m].to(self.device) for m in batch.data}
            ll += self.model.compute_joint_nll(batch)
            nb_batch += 1

        joint_nll = ll / nb_batch
        if hasattr(self, "f"):
            self.f.write(f"\n Joint likelihood : {str(joint_nll)} \n")
