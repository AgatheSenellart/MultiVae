from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import torch
from pythae.models.base.base_utils import ModelOutput
from torchmetrics.classification import MulticlassAccuracy

from multivae.data import MultimodalBaseDataset
from multivae.data.utils import set_inputs_to_device
from multivae.models.base import BaseMultiVAE
from multivae.samplers.base import BaseSampler

from ..base.evaluator_class import Evaluator
from .coherences_config import CoherenceEvaluatorConfig


class CoherenceEvaluator(Evaluator):
    """Class for computing coherences metrics.

    Args:
        model (BaseMultiVAE) : The model to evaluate.
        classifiers (dict) : A dictionary containing the pretrained classifiers to use for the coherence evaluation.
        test_dataset (MultimodalBaseDataset) : The dataset to use for computing the metrics.
        output (str) : The folder path to save metrics. The metrics will be saved in a metrics.txt file.
        eval_config (CoherencesEvaluatorConfig) : The configuration class to specify parameters for the evaluation.
        sampler (BaseSampler) : A custom sampler for computing the joint coherence. If None is provided, samples
            are generated from the prior.
    """

    def __init__(
        self,
        model: BaseMultiVAE,
        classifiers: Dict[str, torch.nn.Module],
        test_dataset: MultimodalBaseDataset,
        output: Optional[str] = None,
        eval_config=CoherenceEvaluatorConfig(),
        sampler: BaseSampler = None,
    ) -> None:
        super().__init__(model, test_dataset, output, eval_config, sampler)
        self.clfs = classifiers
        self.include_recon = eval_config.include_recon
        self.nb_samples_for_joint = eval_config.nb_samples_for_joint
        self.nb_samples_for_cross = eval_config.nb_samples_for_cross
        self.num_classes = eval_config.num_classes
        self.give_details_per_classes = eval_config.give_details_per_class
        assert self.num_classes is not None, "Please provide the number of classes"
        for k in self.clfs:
            self.clfs[k] = self.clfs[k].to(self.device).eval()

    def cross_coherences(self):
        """Computes all the coherences from one subset of modalities to another modality.

        Returns:
            float, float: The cross-coherences metric mean and std
        """
        modalities = list(self.model.encoders.keys())
        accs = []
        accs_per_class = []
        for n in range(1, self.model.n_modalities):
            subsets_of_size_n = combinations(
                modalities,
                n,
            )
            accs.append([])
            accs_per_class.append([])
            for s in subsets_of_size_n:
                s = list(s)
                (
                    subset_dict,
                    mean_acc,
                    mean_acc_per_class,
                ) = self.coherence_from_subset(s, return_accuracies_per_labels=True)
                self.metrics.update(subset_dict)
                accs[-1].append(mean_acc)
                accs_per_class[-1].append(mean_acc_per_class)

        mean_accs = [np.mean(l) for l in accs]
        std_accs = [np.std(l) for l in accs]
        mean_accs_per_class = [np.mean(np.stack(l), axis=0) for l in accs_per_class]

        for i, (m, s) in enumerate(zip(mean_accs, std_accs)):
            self.logger.info(
                "Conditional accuracies for %s modalities : %s +- %s", i + 1, m, s
            )
            self.metrics.update(
                {
                    f"mean_coherence_{i + 1}": m,
                    f"std_coherence_{i + 1}": s,
                }
            )

            if self.give_details_per_classes:
                for c in range(self.num_classes):
                    self.logger.info(
                        "Conditional accuracies for %s modalities in class %s: %s",
                        i + 1,
                        c,
                        mean_accs_per_class[i][c],
                    )
                    self.metrics.update(
                        {
                            f"mean_coherence_{i + 1}_class_{c}": mean_accs_per_class[i][
                                c
                            ],
                        }
                    )

        return mean_accs, std_accs

    def coherence_from_subset(
        self, subset: List[str], return_accuracies_per_labels=False
    ):
        """Compute all the coherences generating from the modalities in subset to a modality
        that is not in subset.

        Args:
            subset (List[str]): The subset of modalities to consider.
            return_accuracies_per_labels (bool): If true, the detailed accuracies per class
                are returned. Default to False.

        Returns:
            dict: The dictionary the detailed coherences of each modalities generated from the subset.
            float: The mean coherence accross subsets.
            (array): If return_accuracies_per_labels is set to True, an array with the detailed accuracies per class is returned.
        """
        pred_mods = [
            m for m in self.model.encoders if (m not in subset) or self.include_recon
        ]

        subset_name = "_".join(subset)

        accuracies_per_class = {
            m: MulticlassAccuracy(num_classes=self.num_classes, average=None).to(
                self.device
            )
            for m in pred_mods
        }

        for batch in self.test_loader:
            if not hasattr(batch, "labels"):
                raise AttributeError(
                    "Cross-modal coherence can not be computed "
                    " on a dataset without labels"
                )
            elif batch.labels is None:
                raise AttributeError(
                    "Cross-modal coherence can not be computed "
                    " on a dataset without labels, but the provided dataset"
                    " has None instead of tensor labels"
                )

            batch = set_inputs_to_device(batch, device=self.device)

            output = self.model.predict(
                batch,
                list(subset),
                pred_mods,
                N=self.nb_samples_for_cross,
                flatten=True,
            )
            for pred_m in pred_mods:
                preds = self.clfs[pred_m](output[pred_m])
                if self.nb_samples_for_cross > 1:
                    labels = torch.stack(
                        [batch.labels] * self.nb_samples_for_cross, dim=0
                    ).reshape(-1, *batch.labels.shape[1:])
                else:
                    labels = batch.labels
                acc = accuracies_per_class[pred_m](preds, labels)

        acc_per_class = {
            f"{subset_name}_to_{m}": accuracies_per_class[m].compute().cpu()
            for m in accuracies_per_class
        }
        acc = {m: acc_per_class[m].mean() for m in acc_per_class}

        self.logger.info("Subset %s accuracies ", subset)
        self.logger.info(str(acc))
        mean_pair_acc = np.mean(list(acc.values()))
        self.logger.info("Mean subset %s accuracies : %s", subset, str(mean_pair_acc))
        mean_acc_per_class = np.mean(np.stack(list(acc_per_class.values())), axis=0)

        if return_accuracies_per_labels:
            return acc, mean_pair_acc, mean_acc_per_class

        else:
            return acc, mean_pair_acc

    def joint_coherence(self):
        """Generate in all modalities from the prior and compute the percentage of samples where all modalities have the same
        labels.

        Returns:
            float: The joint coherence metric
        """
        all_labels = torch.tensor([]).to(self.device)
        samples_to_generate = self.nb_samples_for_joint

        # loop over batches
        while samples_to_generate > 0:
            batch_samples = min(self.batch_size, samples_to_generate)

            if self.sampler is None:
                output_prior = self.model.generate_from_prior(batch_samples)
            else:
                output_prior = self.sampler.sample(batch_samples)

            # set output to device
            output_prior.z = output_prior.z.to(self.device)
            if not output_prior.one_latent_space:
                for m in output_prior.modalities_z:
                    output_prior.modalities_z[m] = output_prior.modalities_z[m].to(
                        self.device
                    )

            # decode
            output_decode = self.model.decode(output_prior)
            labels = []
            for m in output_decode.keys():
                preds = self.clfs[m](output_decode[m])
                labels_m = torch.argmax(preds, dim=1)  # shape (nb_samples_for_joint,1)
                labels.append(labels_m)
            all_same_labels = torch.all(
                torch.stack([l == labels[0] for l in labels]), dim=0
            )
            all_labels = torch.cat((all_labels, all_same_labels.float()), dim=0)
            samples_to_generate -= batch_samples
        joint_coherence = all_labels.mean()

        sampler_name = "prior" if self.sampler is None else self.sampler.name
        self.logger.info(
            "Joint coherence with sampler %s: %s", sampler_name, joint_coherence
        )
        self.metrics.update({f"joint_coherence_{sampler_name}": joint_coherence})
        return joint_coherence

    def eval(self):
        """Compute all cross-modal coherences and the joint coherence.

        Returns:
            ModelOutput: contains all detailed metrics for cross-modal and joint coherence.
        """
        self.cross_coherences()
        self.joint_coherence()

        self.log_to_wandb()

        return ModelOutput(**self.metrics)
