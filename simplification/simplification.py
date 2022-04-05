import cv2
import glob
import os

import numpy as np
import pycocotools.mask as mask_util
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import masks as mask_func
from detectron2.utils.visualizer import GenericMask
from detectron2.utils.visualizer import Visualizer
from simplification.cutil import (
    simplify_coords)
from tqdm import tqdm


def mask_to_polygons(mask):
    mask = np.ascontiguousarray(mask)
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    res = [x + 0.5 for x in res if len(x) >= 6]
    return res, has_holes


def polygons_to_mask(polygons, h, w):
    rle = mask_util.frPyObjects(polygons, h, w)
    rle = mask_util.merge(rle)
    return mask_util.decode(rle)[:, :]


def stack(*args):
    return np.hstack(args)


def simply_segmentations(seg):
    a = []
    for i in seg:
        for j in i:
            a.append(j)

    deger_list = []
    chunk_size = 2
    for j in range(0, len(a), chunk_size):
        deger_list.append(a[j:j + chunk_size])

    deger_list = simplify_coords(deger_list, 10)
    flat_list = []
    for sublist in deger_list:
        for item in sublist:
            flat_list.append(item)
    deger_list = [flat_list]
    return deger_list


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 445
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.MODEL.WEIGHTS = os.path.join("output", "multi_model_mask_rcnn_R_50_C4_3x.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
print(cfg.DATASETS.PROPOSAL_FILES_TRAIN)
predictor = DefaultPredictor(cfg)

for index, img_path in enumerate(tqdm(glob.glob(os.path.join('dataset/tekerlek/tahta', '*')))):
    basename = os.path.basename(img_path)
    im = cv2.imread(img_path)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata={},
                   scale=0.8
                   )
    outputs1 = outputs['instances'].to("cpu")
    for i in range(outputs1.pred_masks.cpu().numpy().shape[0]):
        contours, hierarchy = cv2.findContours(outputs1.pred_masks[i].cpu().numpy().astype("uint8"),
                                               cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        pred, _ = mask_to_polygons(outputs1.pred_masks[i].cpu().numpy())

        for contour in contours:
            convexHull = cv2.convexHull(contour)
            cv2.drawContours(im, [convexHull], -1, (255, 0, 0), 2)
            count = sum([len(listElem) for listElem in contour])

    convexHull = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image_path = f'{index}_{basename}'

    # cv2.imshow("convexHull", convexHull.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image_path = f'{index}_{basename}'
    viz = Visualizer(
        im[:, :, ::-1],
        metadata={},
        scale=0.8,
    )

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    for mask in outputs["instances"].pred_masks.to('cpu').numpy():
        h, w = mask.shape
        pred, _ = GenericMask.mask_to_polygons(self=GenericMask, mask=mask)
        masked = mask_func.polygons_to_bitmask(pred, h, w)
        pred_simply = simply_segmentations(pred)
        masked_simply = mask_func.polygons_to_bitmask(pred_simply, h, w)
        viz.draw_binary_mask(masked_simply)

    simply_count = sum([len(listElem) for listElem in pred_simply])
    print(simply_count, count)

    viz = viz.get_output()
    img = viz.get_image()[:, :, ::-1]
    # cv2.imshow('simply',img)
    image_path = f'{index}_{basename}'
    # cv2.imshow('normal_img',out.get_image()[:, :, ::-1])
    stack = stack(out.get_image()[:, :, ::-1], img, convexHull.get_image()[:, :, ::-1])
    cv2.imshow('stack', stack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
