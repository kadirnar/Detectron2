# Kütüphanelerin Yüklenmesi
import glob

from detectron2.utils.logger import setup_logger
from tqdm import tqdm

setup_logger()
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import os
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances

# Train Veri Setinin Yüklenmesi

register_coco_instances("tekerlek", {}, "./tekerlek/tekerlek_30.json", "car")
register_coco_instances("deneme", {}, "./tekerlek/full.json", "tekerlek")
register_coco_instances("car", {}, "./car/car_100.json", "car")

# register_coco_instances("test_data", {}, "koltuk/train/koltuk.json", "koltuk")

# Cfg Dosyasının Yüklenmesi

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("car",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = os.path.join('output_plate', 'model_final.pth')  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0005  # pick a good LR
cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Resim Boyutu
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Nesne Sayısı
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
"""
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()
torch.save(trainer.model.state_dict(), "./output_plate/model_final.pth")
exit(1)"""

# json görselleştirme

"""
my_dataset_train_metadata = MetadataCatalog.get("deneme")
dataset_dicts = DatasetCatalog.get("deneme")
for d in random.sample(dataset_dicts, 15):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow("img", vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""

# model test

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
predictor = DefaultPredictor(cfg)
my_dataset_train_metadata = MetadataCatalog.get("car")
dataset_dicts = DatasetCatalog.get("car")
for index, img_path in enumerate(tqdm(glob.glob(os.path.join('car', 'train', '*.jpg')))):
    basename = os.path.basename(img_path)
    im = cv2.imread(img_path)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image_path = f'{index}_{basename}'
    # cv2.imwrite(os.path.join('tekerlek', 'train', image_path), out.get_image()[:, :, ::-1])
    cv2.imshow("img", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
