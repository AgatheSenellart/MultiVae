from itertools import combinations
from typing import List

import numpy as np
import torch
from pythae.models.base.base_utils import ModelOutput

from multivae.data import MultimodalBaseDataset
from multivae.data.utils import set_inputs_to_device
from multivae.samplers.base import BaseSampler

from ..base.evaluator_class import Evaluator
from .coherences_config import CoherenceEvaluatorConfig


class CoherenceEvaluator(Evaluator):
    """
    Class for computing coherences metrics.

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
        model,
        classifiers,
        test_dataset,
        output=None,
        eval_config=CoherenceEvaluatorConfig(),
        sampler: BaseSampler = None,
    ) -> None:
        super().__init__(model, test_dataset, output, eval_config, sampler)
        self.clfs = classifiers
        self.include_recon = eval_config.include_recon
        self.nb_samples_for_joint = eval_config.nb_samples_for_joint
        for k in self.clfs:
            self.clfs[k] = self.clfs[k].to(self.device).eval()

    def cross_coherences(self):
        """
        Computes all the coherences from one subset of modalities to another modality.

        Returns:
            float, float: The cross-coherences metric mean and std
        """

        modalities = list(self.model.encoders.keys())
        accs = []
        for n in range(1, self.model.n_modalities):
            subsets_of_size_n = combinations(
                modalities,
                n,
            )
            accs.append([])
            for s in subsets_of_size_n:
                s = list(s)
                subset_dict, mean_acc = self.all_accuracies_from_subset(s)
                self.metrics.update(subset_dict)
                accs[-1].append(mean_acc)
        mean_accs = [np.mean(l) for l in accs]
        std_accs = [np.std(l) for l in accs]

        for i in range(len(mean_accs)):
            self.logger.info(
                f"Conditional accuracies for {i+1} modalities : {mean_accs[i]} +- {std_accs[i]}"
            )
            self.metrics.update(
                {
                    f"mean_coherence_{i+1}": mean_accs[i],
                    f"std_coherence_{i+1}": std_accs[i],
                }
            )
        return mean_accs, std_accs

    def all_accuracies_from_subset(self, subset: List[str]):
        """
        Compute all the coherences generating from the modalities in subset to a modality
        that is not in subset.

        Args:
            subset (List[str]): The subset of modalities to consider.

        Returns:
            dict, float : The dictionary of all coherences from subset, and the mean coherence
        """
        accuracies = {}
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
            pred_mods = [
                m
                for m in self.model.encoders
                if (m not in subset) or self.include_recon
            ]
            output = self.model.predict(batch, list(subset), pred_mods)
            for pred_m in pred_mods:
                preds = self.clfs[pred_m](output[pred_m])
                pred_labels = torch.argmax(preds, dim=1)
                try:
                    accuracies[f"subset_to_{pred_m}"] += torch.sum(
                        pred_labels.cpu() == batch.labels.cpu()
                    )
                except:
                    accuracies[f"subset_to_{pred_m}"] = torch.sum(
                        pred_labels.cpu() == batch.labels.cpu()
                    )

        acc = {k: accuracies[k].cpu().numpy() / self.n_data for k in accuracies}

        self.logger.info(f"Subset {subset} accuracies ")
        self.logger.info(acc.__str__())
        mean_pair_acc = np.mean(list(acc.values()))
        self.logger.info(f"Mean subset {subset} accuracies : " + str(mean_pair_acc))
        return acc, mean_pair_acc

    def joint_coherence(self):
        """
        Generate in all modalities from the prior and compute the percentage of samples where all modalities have the same
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
            f"Joint coherence with sampler {sampler_name}: {joint_coherence}"
        )
        self.metrics.update({f"joint_coherence_{sampler_name}": joint_coherence})
        return joint_coherence

    def eval(self):
        self.cross_coherences()
        self.joint_coherence()

        self.log_to_wandb()

        return ModelOutput(**self.metrics)
