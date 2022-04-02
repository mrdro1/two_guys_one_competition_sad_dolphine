import glob

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import timm
import torch.optim as optim
import torch.nn as nn
import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as LP
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import InferenceModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import BaseConfigClass
from data import HappyWhaleDataset

TEST_SIZE = 0.05


def load_df(fn):
    df = pd.read_csv(fn)
    df['label'] = df.groupby('individual_id').ngroup()
    return df


def train_test_split_impl(df, seed):
    return train_test_split(df, test_size=TEST_SIZE, random_state=seed)


class PytorchMetricLearningTask:
    def __init__(self, config: BaseConfigClass, device):
        self.config = config
        self.device = device
        self.inference_model = None
        self.threshold = None

    def setup_task(self, train_dataset, val_dataset, train_df, val_df, combined_df):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_df = train_df
        self.val_df = val_df
        self.combined_df = combined_df
        self.dataset_dict = {"train": train_dataset, "val": val_dataset}

        self.trunk, self.embedder = self.get_model_()
        self.loss = self.get_loss_()
        self.hooks = self.get_hooks_handler_()
        self.tester, self.end_of_epoch_hook = self.get_tester_()
        self.trunk_optimizer, self.trunk_schedule = self.get_optimizer_(self.trunk)
        self.embedder_optimizer, self.embedder_schedule = self.get_optimizer_(self.embedder)
        self.loss_optimizer, self.loss_schedule = self.get_optimizer_(self.trunk)
        self.trainer = self.get_trainer_()

    def train(self, n_epoch):
        print(self.trainer, n_epoch)
        self.trainer.train(n_epoch)

    def load_models(self, checkpoint_name):
        '''
        :param checkpoint_name: best, ??
        '''
        best_trunk_weights = glob.glob(f'../models/{self.config.model_name}/trunk_{checkpoint_name}*.pth')[0]
        self.trunk.load_state_dict(torch.load(best_trunk_weights))

        best_embedder_weights = glob.glob(f'../models/{self.config.model_name}/embedder_{checkpoint_name}*.pth')[0]
        self.embedder.load_state_dict(torch.load(best_embedder_weights))

    def predict(self, database: HappyWhaleDataset, x_dataset: HappyWhaleDataset) -> pd.DataFrame:
        if self.threshold is None:
            raise Exception('Threshold for prediction not defined')
        self.setup_inference_(database)
        distances, indices, _ = self.infer_(x_dataset)

        new_whale_idx = -1

        combined_idx_lookup = self.combined_df['individual_id'].copy().to_dict()
        combined_idx_lookup[-1] = 'new_individual'

        prediction_list = []
        for i in range(len(distances)):
            pred_knn_idx = indices[i, :].copy()
            insert_idx = np.where(distances[i, :] > self.threshold)

            if insert_idx[0].size != 0:
                pred_knn_idx = np.insert(pred_knn_idx, np.min(insert_idx[0]), new_whale_idx)

            predicted_label_list = []

            for predicted_idx in pred_knn_idx:
                predicted_label = combined_idx_lookup[predicted_idx]
                if len(predicted_label_list) == 5:
                    break
                if (predicted_label == 'new_individual') | (predicted_label not in predicted_label_list):
                    predicted_label_list.append(predicted_label)

            prediction_list.append(predicted_label_list)

        prediction_df = pd.DataFrame(prediction_list)
        return prediction_df

    def validation_step(self) -> list:
        self.setup_inference_(self.train_dataset)
        threshold, results_df = self.evaluate_threshold()
        self.threshold = threshold
        val_metrics = results_df.loc[:5, 'map5']

        return list(val_metrics)

    def evaluate_threshold(self):
        val_distances, val_indices, val_labels = self.infer_(self.val_dataset, return_labels=True)

        new_whale_idx = -1

        train_labels = self.train_df['individual_id'].unique()
        train_idx_lookup = self.train_df['individual_id'].copy().to_dict()
        train_idx_lookup[-1] = 'new_individual'

        valid_class_lookup = self.val_df.set_index('label')['individual_id'].copy().to_dict()

        thresholds = [np.quantile(val_distances, q=q) for q in np.arange(0, 1.0, 0.01)]
        results = []
        for threshold in tqdm(thresholds):
            running_map = 0
            for i in range(len(val_distances)):
                pred_knn_idx = val_indices[i, :].copy()
                insert_idx = np.where(val_distances[i, :] > threshold)
                if insert_idx[0].size != 0:
                    pred_knn_idx = np.insert(pred_knn_idx, np.min(insert_idx[0]), new_whale_idx)

                predicted_label_list = []
                for predicted_idx in pred_knn_idx:
                    predicted_label = train_idx_lookup[predicted_idx]
                    if len(predicted_label_list) == 5:
                        break
                    if (predicted_label == 'new_individual') | (predicted_label not in predicted_label_list):
                        predicted_label_list.append(predicted_label)

                gt = valid_class_lookup[val_labels[i]]
                if gt not in train_labels:
                    gt = "new_individual"

                precision_vals = []
                for j in range(5):
                    if predicted_label_list[j] == gt:
                        precision_vals.append(1 / (j + 1))
                    else:
                        precision_vals.append(0)

                running_map += np.max(precision_vals)

            results.append([threshold, running_map / len(val_distances)])

        results_df = pd.DataFrame(results, columns=['threshold', 'map5'])
        results_df = results_df.sort_values(by='map5',
                                            ascending=False).reset_index(drop=True)
        threshold = results_df.loc[0, 'threshold']
        return threshold, results_df

    def setup_inference_(self, dataset):
        self.inference_model = InferenceModel(
            trunk=self.trunk,
            embedder=self.embedder,
            normalize_embeddings=True,
        )
        self.inference_model.train_knn(dataset)

    def infer_(self, x_dataset, return_labels=False):
        x_dataloader = DataLoader(x_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.n_worker,
                                     pin_memory=True)
        distance_list = []
        indices_list = []
        labels_list = []
        for sample in tqdm(x_dataloader):
            if return_labels:
                images, labels = sample
            else:
                images = sample
            distances, indices = self.inference_model.get_nearest_neighbors(images, k=self.config.n_neighbours)
            distance_list.append(distances)
            indices_list.append(indices)
            if return_labels:
                labels_list.append(labels)

        distances = torch.cat(distance_list, dim=0).cpu().numpy()
        indices = torch.cat(indices_list, dim=0).cpu().numpy()
        labels = torch.cat(labels_list, dim=0).cpu().numpy() if return_labels else None
        return distances, indices, labels

    def get_model_(self):
        trunk = timm.create_model(self.config.model_name, pretrained=True)
        trunk.classifier = common_functions.Identity()
        trunk = trunk.to(self.device)

        embedder = nn.Linear(self.config.backbone_output_size, self.config.embedding_size).to(self.device)
        return trunk, embedder

    def get_optimizer_(self, model):
        optimizer = optim.SGD(model.parameters(), lr=self.config.model_lr, momentum=self.config.momentum)
        schedule = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.model_lr,
            total_steps=self.config.total_steps(self.train_dataset),
            pct_start=self.config.pct_start
        )
        return optimizer, schedule

    def get_loss_(self):
        if self.config.loss_name == 'ArcFaceLoss':
            loss = losses.ArcFaceLoss(num_classes=self.config.num_classes, embedding_size=self.config.embedding_size).to(self.device)
        else:
            raise Exception(f'Invalid loss name: {self.config.loss_name}')

        return loss

    def get_hooks_handler_(self):
        record_keeper, _, _ = LP.get_record_keeper(self.config.log_dir)
        hooks = LP.get_hook_container(record_keeper, primary_metric='mean_average_precision')
        return hooks

    def get_tester_(self):
        tester = testers.GlobalEmbeddingSpaceTester(
            end_of_testing_hook=self.hooks.end_of_testing_hook,
            accuracy_calculator=AccuracyCalculator(
                include=['mean_average_precision'],
                device=torch.device("cpu"),
                k=5),
            dataloader_num_workers=self.config.n_worker,
            batch_size=self.config.batch_size
        )

        end_of_epoch_hook = self.hooks.end_of_epoch_hook(
            tester,
            self.dataset_dict,
            self.config.model_dir,
            test_interval=1,
            patience=self.config.patience,
            splits_to_eval=[('val', ['train'])]
        )
        return tester, end_of_epoch_hook

    def get_trainer_(self):
        trainer = trainers.MetricLossOnly(
            models={"trunk": self.trunk, "embedder": self.embedder},
            optimizers={"trunk_optimizer": self.trunk_optimizer, "embedder_optimizer": self.embedder_optimizer,
                        "metric_loss_optimizer": self.loss_optimizer},
            batch_size=self.config.batch_size,
            loss_funcs={"metric_loss": self.loss},
            mining_funcs={},
            dataset=self.train_dataset,
            dataloader_num_workers=self.config.n_worker,
            end_of_epoch_hook=self.end_of_epoch_hook,
            lr_schedulers={
                'trunk_scheduler_by_iteration': self.trunk_schedule,
                'embedder_scheduler_by_iteration': self.embedder_schedule,
                'metric_loss_scheduler_by_iteration': self.loss_schedule,
            }
        )
        return trainer
