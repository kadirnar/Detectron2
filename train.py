# library imports

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer


# train and val register coco instances
def register_coco(name, json_file, image_root):
    """
    Register a COCO dataset to detectron2.
    Args:
        name: the name of the dataset
        json_file: the path to the COCO annotation file
        image_root: the path to the COCO images
    """
    # register an empty object detection dataset
    register_coco_instances(name, {}, json_file, image_root)


# detectron2 train config setup using object object oriented

def setup_cfg(model, data_train, num_gpus, output_dir, batch_size, num_iter, lr, num_classes):
    """

    Args:
        num_iter: number of iterations
        lr: learning rate
        num_classes: number of classes
        batch_size: per batch size of images to process in training process
        model: model zoo name of the model
        data_train: coco dataset name
        num_gpus: driver number of gpus to use for training
        output_dir: file path to save the model

    """

    cfg = get_cfg()
    # set the output directory
    cfg.OUTPUT_DIR = output_dir
    # set the number of gpus
    cfg.MODEL.DEVICE = "cuda" if num_gpus > 0 else "cpu"
    # set the model name
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    # set the dataset name
    cfg.DATASETS.TRAIN = (data_train,)
    # set the data augmentation
    cfg.DATALOADER.NUM_WORKERS = num_gpus
    # set the number of images per batch
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    # set the number of iterations
    cfg.SOLVER.MAX_ITER = num_iter
    # set the number of iterations per epoch
    cfg.SOLVER.STEPS = (100, 200)
    # set the learning rate
    cfg.SOLVER.BASE_LR = lr
    # set the momentum
    cfg.SOLVER.MOMENTUM = 0.9
    # set the weight decay
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    # set the gamma
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = 50
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    return cfg


if __name__ == "__main__":
    # dataset path and name
    register_coco("coco_train", "./coco/annotations/instances_train2014.json", "./coco/train2014")
    # model parameters
    model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    data_train, = "coco_train"
    num_gpus = 4
    output_dir = "./output"
    batch_size = 16
    num_iter = 1000
    lr = 0.0001
    num_classes = 80
    cfg = setup_cfg(model, data_train, num_gpus, output_dir, batch_size, num_iter, lr, num_classes)
    # train the model
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    exit(0)
