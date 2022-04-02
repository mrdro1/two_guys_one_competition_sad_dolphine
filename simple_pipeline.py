import numpy as np
import pandas as pd
import torch

import lib
import data
from configs import baseline_config as config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
np.random.seed(config.seed)
torch.manual_seed(config.seed)

# Data preparations
df = lib.load_df(config.train_csv)
print(len(df))
train_df, val_df = lib.train_test_split_impl(df, config.seed)
print(len(train_df), len(val_df))
# TODO assert shape
train_dataset = data.HappyWhaleDataset(df=train_df, image_dir=config.train_dir, return_labels=True)
val_dataset = data.HappyWhaleDataset(df=val_df, image_dir=config.train_dir, return_labels=True)

task = lib.PytorchMetricLearningTask(config, device)
task.setup_task(train_dataset, val_dataset, train_df, val_df, df)

task.train(config.n_epochs)

task.load_models('best')
val_metric = task.validation_step()
print(val_metric)

test_df = pd.read_csv(config.test_csv)
train_dataset = data.HappyWhaleDataset(df=df, image_dir=config.train_dir, return_labels=True)
test_dataset = data.HappyWhaleDataset(df=test_df, image_dir=config.train_dir, return_labels=True)

y_predictions = task.predict(train_dataset, test_dataset)

y_predictions['predictions'] = y_predictions[0].astype(str) + ' ' + y_predictions[1].astype(str) + ' ' +\
                               y_predictions[2].astype(str) + ' ' + y_predictions[3].astype(str) + ' ' +\
                               y_predictions[4].astype(str)

submission = pd.read_csv(config.test_csv)
submission['predictions'] = y_predictions['predictions']
submission.to_csv('submission.csv', index=False)