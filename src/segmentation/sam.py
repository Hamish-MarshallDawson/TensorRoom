from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import yaml

ONTOLOGY_TERMS = {
    "table": "table",
    "coffee table": "table",
    "window": "window",
    "lamp": "lamp",
    "ottoman": "ottoman",
}


def load_config():
    with open(Path(__file__).parent.parent.parent / "config.yaml", "r") as f:
        return yaml.safe_load(f)

def show_mask(mask, ax, color=np.array([30/255, 144/255, 255/255, 0.6])):
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="yellow", facecolor="none", lw=2))


def get_label(detections, index, ontology_terms):
    labels_by_class_id = list(ontology_terms.keys())
    label = "object"

    class_ids = getattr(detections, "class_id", None)
    if class_ids is not None and len(class_ids) > index:
        class_id = int(class_ids[index])
        if 0 <= class_id < len(labels_by_class_id):
            label = labels_by_class_id[class_id]

    confidence = getattr(detections, "confidence", None)
    if confidence is not None and len(confidence) > index:
        return f"{label} {float(confidence[index]):.2f}"
    return label


def show_label(box, label, ax):
    x0, y0 = box[0], box[1]
    ax.text(
        x0,
        max(0, y0 - 5),
        label,
        color="black",
        fontsize=10,
        bbox=dict(facecolor="yellow", alpha=0.7, edgecolor="none", pad=2),
    )

def main():
    cfg = load_config()
    image = cv2.imread("../../data/images/output.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model = GroundedSAM(
        ontology=CaptionOntology(ONTOLOGY_TERMS),
        box_threshold=cfg.get("box_threshold"),
        text_threshold=cfg.get("text_threshold"),
    )
    results = model.predict(image)

    if len(results.xyxy) == 0:
        print("No detections")
    else:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        ax = plt.gca()
        for i in range(len(results.xyxy)):
            show_mask(results.mask[i], ax)
            show_box(results.xyxy[i], ax)
            show_label(results.xyxy[i], get_label(results, i, ONTOLOGY_TERMS), ax)
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    main()