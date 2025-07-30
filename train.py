import torch
assert torch.__version__.startswith("1.10")

import os

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances

# Clear data to avoid interference
DatasetCatalog.clear()
MetadataCatalog.clear()


#----------------------------
# Register train and val sets
#----------------------------

register_coco_instances("dataset_train", {},
                        "dataset/annotations/train_annotations.json",
                        "dataset/train")

register_coco_instances("dataset_test", {},
                        "dataset/annotations/val_annotations.json",
                        "dataset/val")


#------------------------------
# Configure training parameters
#------------------------------

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("dataset_train",)
cfg.DATASETS.TEST = ("dataset_val",)

cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20


#-------------------------------------------------------------------------------
# Create custom image augmentation pipeline to artificially increase sample size
#-------------------------------------------------------------------------------

class AugmentedTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        augmentation_list = [
            T.Resize((1200, 900)),
            T.RandomBrightness(0.5, 2),
            T.RandomContrast(0.5, 2),
            T.RandomSaturation(0.5, 2),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomLighting(0.7),
        ]
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentation_list)
        return build_detection_train_loader(cfg, mapper=mapper)

#make the output directory
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


#----------------
# Train the model
#----------------

trainer = AugmentedTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


#-------------------
# Save trained model
#-------------------

checkpointer = DetectionCheckpointer(trainer.model, save_dir="models/")
checkpointer.save("model_0")

# Save config file
f = open('models/model_0_cfg.yaml','w')
f.write(cfg.dump())
f.close()

