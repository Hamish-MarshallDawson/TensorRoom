from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
import matplotlib.pyplot as plt
import numpy as np
import cv2

def show_mask(mask, ax, color=np.array([30/255, 144/255, 255/255, 0.6])):
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor='none', lw=2))

image = cv2.imread("../../data/images/output.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

model = GroundedSAM(
    ontology=CaptionOntology({
        "coffee table": "table",
        "living room table": "table",
        "wooden table": "table",
        "center table": "table",
        "table": "table",
    }),
    box_threshold=0.2,
    text_threshold=0.15,
)
results = model.predict(image)

if len(results.xyxy) == 0:
    print("No detections")
else:
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(results.mask[0], plt.gca())
    show_box(results.xyxy[0], plt.gca())
    plt.axis("off")
    plt.show()
