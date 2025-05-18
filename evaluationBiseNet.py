# MEAN INTERSECTION OVER UNION
# The Intersection over Union (IoU) metric, also referred to as the Jaccard index,
# quantifies the percent overlap between the target mask and the prediction output.

import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from models.deeplabv2.deeplabv2 import get_deeplab_v2
import datasets.cityscapes as cityscapes
from fvcore.nn import FlopCountAnalysis, flop_count_table

from models.bisenet.build_bisenet import BiSeNet


#DICE The IoU score is calculated for each class separately and then averaged over all classes to provide a global,
#mean IoU score of our semantic segmentation prediction.

id_to_class_name = {
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'traffic light',
    7: 'traffic sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle'
}


def calculate_mean_iou_per_class(preds, gts, num_classes=19):
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)

    for pred, gt in zip(preds, gts):
        for cls in range(num_classes):
            pred_mask = (pred == cls)
            gt_mask = (gt == cls)

            inter = np.logical_and(pred_mask, gt_mask).sum()
            uni = np.logical_or(pred_mask, gt_mask).sum()

            intersection[cls] += inter
            union[cls] += uni

    class_iou = intersection / (union + 1e-10)
    mean_iou = np.nanmean(class_iou)  # skip NaNs if any class never occurs

    return mean_iou, class_iou



def evaluate_model(model, outputs, masks, input_size=(224, 224), iterations=1000, device='cpu'):
    print("\n=== MODEL EVALUATION ===")
    model.eval()
    model.to(device)

    outputs_np = outputs.cpu().detach().numpy()
    masks_np = masks.cpu().detach().numpy()

    preds = [np.argmax(output, axis=0) for output in outputs_np]
    gts = [mask[0] for mask in masks_np]

    mean_iou, class_ious = calculate_mean_iou_per_class(preds, gts, num_classes=19)

    print(f"\nMean IoU (per class average): {mean_iou:.4f}")
    print("IoU per classe:")
    for cls_id, iou in enumerate(class_ious):
        class_name = id_to_class_name.get(cls_id, f"Class {cls_id}")
        iou_display = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
        print(f"{class_name:>15}: {iou_display}")


    # Latenza & FPS
    height, width = input_size
    image = torch.rand(1, 3, height, width).to(device)

    latency = []
    fps_list = []

    for _ in range(iterations):
        start = time.time()
        with torch.no_grad():
            _ = model(image)
        end = time.time()

        latency_i = end - start
        latency.append(latency_i)
        fps_list.append(1 / latency_i)

    mean_latency = np.mean(latency) * 1000
    std_latency = np.std(latency) * 1000
    mean_fps = np.mean(fps_list)
    std_fps = np.std(fps_list)

    print(f"Mean Latency: {mean_latency:.2f} ms")
    print(f"Std Latency: {std_latency:.2f} ms")
    print(f"Mean FPS: {mean_fps:.2f}")
    print(f"Std FPS: {std_fps:.2f}")

    # FLOPs
    image_flop = torch.zeros((1, 3, height, width)).to(device)
    flops = FlopCountAnalysis(model, image_flop)
    print(flop_count_table(flops))

  
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   
    val_csv = '/content/SemSeg_MLDL25/val_annotations.csv'
    base_path = './Cityscapes/Cityscapes/Cityspaces'
    val_dataset = cityscapes.CityScapes(
        annotations_file=val_csv,
        root_dir=base_path,
        transform=cityscapes.transform['image'],
        target_transform=cityscapes.transform['mask']
    )

    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)
    model = BiSeNet(num_classes=19, context_path='resnet18').to(device)
    model.load_state_dict(torch.load('/content/SemSeg_MLDL25/checkpoints/final_model_lr2.5e-05_bs4.pth', map_location=device))


   
    model.eval()


    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)


   
    evaluate_model(model, outputs, targets, input_size=(512, 1024), iterations=100, device=device)
