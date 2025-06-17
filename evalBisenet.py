import os
import torch
import numpy as np
import time
import csv
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis, flop_count_table
from models.bisenet.build_bisenet import BiSeNet
import datasets.cityscapes as cityscapes

# Mappa ID classe → nome
id_to_class_name = {
    0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
    5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation',
    9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
    14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
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

def evaluate_model(model, val_loader, input_size=(512, 1024), iterations=100, device='cpu'):
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
    total_flops = flops.total()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return mean_iou, class_ious, mean_latency, std_latency, mean_fps, std_fps, total_flops, total_params, trainable_params

if _name_ == "_main_":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_csv = './SemSeg_MLDL25/val_annotations.csv'
    base_path = './Cityscapes'
    val_dataset = cityscapes.CityScapes(
        annotations_file=val_csv,
        root_dir=base_path,
        transform=cityscapes.transform['image'],
        target_transform=cityscapes.transform['mask']
    )

    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)

    checkpoints_dir = './checkpoints_bisenet'
    output_csv = 'eval_bisenet_results.csv'

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['Checkpoint', 'Mean IoU', 'Mean Latency (ms)', 'Std Latency (ms)', 'Mean FPS', 'Std FPS', 'FLOPs', 'Total Params', 'Trainable Params']
        header.extend([id_to_class_name[i] for i in range(19)])
        writer.writerow(header)

        for ckpt_file in os.listdir(checkpoints_dir):
            if ckpt_file.endswith('.pth'):
                ckpt_path = os.path.join(checkpoints_dir, ckpt_file)
                print(f"Evaluating {ckpt_file}...")

                model = BiSeNet(num_classes=19, context_path='resnet18').to(device)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))

                mean_iou, class_ious, mean_latency, std_latency, mean_fps, std_fps, total_flops, total_params, trainable_params = evaluate_model(
                    model, val_loader, input_size=(512, 1024), iterations=100, device=device)

                row = [ckpt_file, f"{mean_iou:.4f}", f"{mean_latency:.2f}", f"{std_latency:.2f}", f"{mean_fps:.2f}", f"{std_fps:.2f}", f"{total_flops:.2e}", f"{total_params}", f"{trainable_params}"]
                row.extend([f"{iou:.4f}" for iou in class_ious])
                writer.writerow(row)

    print(f"Evaluation completed. Results saved to {output_csv}"
