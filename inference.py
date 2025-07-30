import os
import cv2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from matplotlib import pyplot as plt


# Define hold types
classes = ['jug', 'crimp', 'pinch', 'pocket', 'sloper', 'edge', 'jib', 'volume',
           'slopey_jug', 'slopey_crimp', 'slopey_pinch', 'slopey_pocket', 'slopey_edge',
           'slopey_jib', 'slopey_volume', 'crimpy_jug', 'crimpy_pinch', 'crimpy_pocket',
           'crimpy_edge', 'crimpy_jib']

#display the custom classes
MetadataCatalog.get("dataset_train").set(thing_classes=classes)
microcontroller_metadata = MetadataCatalog.get("dataset_train")


# Configure the inference parameters
cfg = get_cfg()
cfg.merge_from_file("models/model_0_cfg.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Set threshold for this model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
cfg.MODEL.WEIGHTS = "models/model_0.pth"
cfg.DATASETS.TEST = ("dataset_test", )

predictor = DefaultPredictor(cfg)

# Run inference on the selected image
img = cv2.imread("dataset/train/image23.jpg")
outputs = predictor(img)

# Visualize results
v = Visualizer(img[:, :, ::-1], metadata=microcontroller_metadata, scale=0.8,
               instance_mode=ColorMode.IMAGE_BW)  # removes the colors of unsegmented pixels
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

plt.figure(figsize=(25, 25))
plt.imshow(v.get_image())
plt.axis('off')
plt.show()