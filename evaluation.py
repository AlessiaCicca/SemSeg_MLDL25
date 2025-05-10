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


def calculate_iou(predicted_mask, target_mask):
    intersection = np.logical_and(predicted_mask, target_mask).sum()
    union = np.logical_or(predicted_mask, target_mask).sum()
    iou = intersection / (union + 1e-10)
    return iou


def evaluate_model(model, outputs, masks, input_size=(224, 224), iterations=1000, device='cpu'):
    print("\n=== MODEL EVALUATION ===")
    model.eval()
    model.to(device)

    outputs_np = outputs.cpu().detach().numpy()
    masks_np = masks.cpu().detach().numpy()

    iou_scores = []
    for i in range(len(outputs_np)):
        predicted_mask = np.argmax(outputs_np[i], axis=0)
        target_mask = masks_np[i, 0]
        iou = calculate_iou(predicted_mask, target_mask)
        iou_scores.append(iou)

    mean_iou = np.mean(iou_scores)
    print(f"Mean IoU: {mean_iou:.4f}")

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


if __name__ == "_main_":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   
    val_csv = '/content/SemSeg_MLDL25/val_annotations.csv'
    base_path = '/tmp/Cityscapes/Cityscapes/Cityspaces'

    val_dataset = cityscapes.CityScapes(
        annotations_file=val_csv,
        root_dir=base_path,
        transform=cityscapes.transform['image'],
        target_transform=cityscapes.transform['mask']
    )

    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)

   
    model = get_deeplab_v2(num_classes=19).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))

   
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

   
    evaluate_model(model, outputs, targets, input_size=(512, 1024), iterations=100, device=device)