import numpy as np
import torch
from torch.utils.data import DataLoader

from multivae.data import MultimodalBaseDataset
from multivae.data.utils import set_inputs_to_device
from multivae.models.base import BaseMultiVAE, ModelOutput

from ..base.evaluator_class import Evaluator
from .clustering_config import ClusteringConfig


class Clustering(Evaluator):
    """Module to perform clustering in the latent space.
    As of now, it is only supported for the joint representation of the data.
    The eval() function fits a k-means model on the training embeddings, then uses this model
    to classify the test_samples and returns a k-means accuracy for this prediction.

    Args:
        model (BaseMultiVAE) : The model to evaluate.
        test_dataset (MultimodalBaseDataset) : The dataset to use for computing the metrics.
        train_dataset (MultimodalBaseDataset): The training dataset to fit the k-means.
        output (str) : The folder path to save metrics. The metrics will be saved in a metrics.txt
            file.
        eval_config (EvaluatorConfig) : The configuration class to specify parameters for the
            evaluation.

    """

    def __init__(
        self,
        model: BaseMultiVAE,
        test_dataset: MultimodalBaseDataset,
        train_dataset: MultimodalBaseDataset,
        output: str = None,
        eval_config=ClusteringConfig(),
    ) -> None:
        super().__init__(model, test_dataset, output, eval_config)

        self.num_samples_for_fit = eval_config.num_samples_for_fit
        self.n_fits = eval_config.number_of_runs
        self.use_mean = eval_config.use_mean
        if eval_config.clustering_method == "kmeans":
            try:
                from sklearn.cluster import KMeans
            except:
                raise ModuleNotFoundError(
                    "scikit-learn must be installed to perform clustering. Run `pip install scikit-learn` to install it "
                )
            self.clustering = KMeans(n_clusters=eval_config.n_clusters, max_iter=300)

            self.train_dataset = train_dataset

    def fit_clustering(self, mods="all"):
        # compute all training embeddings
        dl = DataLoader(self.train_dataset, self.batch_size, shuffle=True)

        list_z = []
        n_samples = 0
        labels = []
        for inputs in dl:
            if (
                self.num_samples_for_fit is not None
                and n_samples > self.num_samples_for_fit
            ):
                break
            inputs = set_inputs_to_device(inputs, self.device)
            with torch.no_grad():
                list_z.append(
                    self.model.encode(inputs, mods, return_mean=self.use_mean).z
                )
            if inputs.labels is not None:
                labels.append(inputs.labels)

        if len(labels) > 0:
            labels = torch.cat(labels).cpu().numpy()
            labels.dtype = np.int64
        all_z = torch.cat(list_z).cpu().numpy()
        clusters_labels = self.clustering.fit_predict(all_z)
        # Get the majority label for each cluster
        self.labels_dict = {str(m): m for m in np.unique(clusters_labels)}
        if len(labels) == len(clusters_labels):
            for c in np.unique(clusters_labels):
                maj_value = np.bincount(labels[clusters_labels == c]).argmax()
                self.labels_dict[str(c)] = maj_value

    def cluster_accuracy(self, mods="all"):
        mean_acc = []
        for i in range(self.n_fits):
            self.fit_clustering(mods)

            # Cluster the test dataset
            acc = 0
            n_samples = 0
            for inputs in self.test_loader:
                inputs = set_inputs_to_device(inputs, self.device)
                with torch.no_grad():
                    z = self.model.encode(inputs, mods, return_mean=self.use_mean).z
                clabels = self.clustering.predict(z.cpu().numpy())
                labels = np.array([self.labels_dict[str(c)] for c in clabels])
                true_labels = inputs.labels.cpu().numpy()
                acc += np.sum(labels == true_labels)
                n_samples += len(z)

            accuracy = acc / n_samples
            mean_acc.append(accuracy)
        accuracy = np.mean(mean_acc)
        self.metrics["cluster_accuracy"] = accuracy
        self.logger.info(f"Cluster accuracy is {accuracy}")
        return ModelOutput(cluster_accuracy=accuracy)

    def eval(self):
        output = self.cluster_accuracy("all")
        self.log_to_wandb()
        return output
