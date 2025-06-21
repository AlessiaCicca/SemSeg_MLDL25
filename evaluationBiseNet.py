import os
import torch
import numpy as np
import time
import csv
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis, flop_count_table
from models.bisenet.build_bisenet import BiSeNet
import datasets.cityscapes as cityscapes

# Dictionary mapping class ID (integer) to its corresponding semantic class name.
id_to_class_name = {
    0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
    5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation',
    9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
    14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
}

def calculate_mean_iou_per_class(preds, gts, num_classes=19):

    """
    Computes the mean Intersection over Union (mIoU) and per-class IoU.

    Args:
        preds : List of predicted label maps (H, W) with class indices.
        gts : List of ground truth label maps (H, W) with class indices.
        num_classes : Number of valid classes (default: 19).

    Returns:
        tuple:
            mean_iou : Mean IoU across all classes.
            class_iou : Array containing the IoU for each class.

    For each prediction and ground truth pair, this function computes the intersection 
    and union for each class across all images. The IoU for each class is calculated as 
    the ratio of intersection over union. The mean IoU is the average of the per-class IoUs.
    
    """
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
    mean_iou = np.nanmean(class_iou)

    return mean_iou, class_iou

def evaluate_model(model, val_loader, input_size=(512, 1024), iterations=100, device='cpu'):

    """
    Evaluates the model on a validation dataset and reports key performance metrics.

    Args:
        model : The neural network model to be evaluated.
        val_loader : DataLoader providing validation data.
        input_size : Input dimensions (height, width) for latency and FLOPs measurement.
        iterations : Number of iterations for latency and FPS benchmarking.
        device : The device on which computations are performed (CPU).

    This function sets the model to evaluation mode and disables gradient computation.
    It iterates through the validation data, collecting predictions and ground truths to compute:
      - mean Intersection over Union (mIoU)
      - per-class IoU
    It also benchmarks:
      - latency and frames per second (FPS) over the specified number of iterations
      - FLOPs (floating-point operations) using FlopCountAnalysis
      - total and trainable parameter counts
    
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            gts = targets.squeeze(1).cpu().numpy()
            all_preds.extend(preds)
            all_gts.extend(gts)

    mean_iou, class_ious = calculate_mean_iou_per_class(all_preds, all_gts, num_classes=19)

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

    # FLOPs
    image_flop = torch.zeros((1, 3, height, width)).to(device)
    flops = FlopCountAnalysis(model, image_flop)
 

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n=== MODEL EVALUATION RESULTS ===")
    print(f"Mean IoU (average over classes): {mean_iou:.4f}")
    print("IoU per classe:")
    for cls_id, iou in enumerate(class_ious):
        class_name = id_to_class_name.get(cls_id, f"Class {cls_id}")
        iou_display = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
        print(f"{class_name:>15}: {iou_display}")

    print(f"\nMean Latency: {mean_latency:.2f} ms")
    print(f"Std Latency: {std_latency:.2f} ms")
    print(f"Mean FPS: {mean_fps:.2f}")
    print(f"Std FPS: {std_fps:.2f}")

    print(f"\nFLOPs tables: {flop_count_table(flops)}")
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
    model.load_state_dict(torch.load('/content/SemSeg_MLDL25/best_model.pth', map_location=device))
    model.eval()

    
    evaluate_model(model, val_loader, input_size=(512, 1024), iterations=100, device=device)
