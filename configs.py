from dataclasses import dataclass
import abc
# TODO разнести классы, чтобы конфиг треньки был отдельным файлом
@dataclass
class BaseConfigClass:
    # files
    train_dir = '../input/happywhale-cropped-dataset-yolov5-ds/train_images'
    test_dir = '../input/happywhale-cropped-dataset-yolov5-ds/test_images'
    train_csv = '../input/happy-whale-and-dolphin/train.csv'
    test_csv = '../input/happy-whale-and-dolphin/sample_submission.csv'
    # model
    model_name: str
    num_classes: int
    backbone_output_size: int
    embedding_size: int
    # loss
    loss_name: str
    # training
    seed: int
    n_epochs: int
    batch_size: int
    model_lr: float
    momentum: float
    pct_start: float
    # search
    n_neighbours: int
    # data load
    n_worker: int
    # log
    patience: int

    def __post_init__(self):
        self.log_dir = "../logs/{}".format(self.model_name)
        self.model_dir = "../models/{}".format(self.model_name)

    def total_steps(self, dataset):
        return self.n_epochs * int(len(dataset) / self.batch_size)


baseline_config = BaseConfigClass(
    model_name='tf_efficientnet_b3_ns',
    num_classes=15587,
    backbone_output_size=1536,
    embedding_size=512,

    loss_name='ArcFaceLoss',

    seed=666,
    n_epochs=15,
    batch_size=24,

    model_lr=1e-3,
    momentum=0.9,
    pct_start=0.3,

    n_neighbours=1000,

    n_worker=2,

    patience=5
)


