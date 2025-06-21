
#TRAIN 2A
import os
import zipfile
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler 
import numpy as np
from models.deeplabv2.deeplabv2 import get_deeplab_v2
import datasets.cityscapes as cityscapes

#GradScaler helps make mixed precision training more stable and efficient by automatically scaling the loss and gradients to prevent
#numerical issues when using 16-bit floating point values. Mixed precision training means using a combination of 16-bit floating point (float16) 
#and 32-bit floating point (float32) numbers during neural network training, instead of using only 32-bit floats, to faster training, lower memory usage

scaler = GradScaler()  

#-------------------------
#       COMPUTE mIou 
#-------------------------

#This function calculates the mean Intersection over Union (mIoU):
def compute_miou(preds, labels, num_classes=19, ignore_index=255):
    ious = []

#.cpu().numpy() is used to move a PyTorch tensor from GPU to CPU and convert it into a NumPy array, 
#so it can be used with NumPy operations, which require CPU-based data.

    preds = preds.cpu().numpy()    #predicted value
    labels = labels.cpu().numpy()  #ground truth masks

#Looping over every semantic class

    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = (preds == cls)
        label_inds = (labels == cls)
        
#The metric is defined by the overlap between the predicted segmentation and the ground truth, 
#divided by the total area covered by the union of the two, so we compute intersection and union

        intersection = np.logical_and(pred_inds, label_inds).sum()
        union = np.logical_or(pred_inds, label_inds).sum()
        if union == 0:
            continue
        iou = intersection / union
        ious.append(iou)
        
    if len(ious) == 0:
        return 0.0        
#return the mean between the Iou of each class
    return np.mean(ious)


#-------------------------
#       TRAINING 
#-------------------------

def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()  #Sets the model to training mode
    running_loss, correct, total = 0.0, 0, 0    #initialization of the values

    #Iterates over batches from the training dataset.
    for inputs, targets in train_loader:
        #Moves the images and masks to the GPU 
        inputs, targets = inputs.to(device), targets.to(device)
        #Clears previous gradients from the optimizer
        optimizer.zero_grad()
        #Enables mixed precision training, which uses float16 for performance and memory efficiency on GPU (DUE TO COMPUTATIONAL ISSUES)
        with autocast('cuda'):
            outputs = model(inputs)
            #Computes the loss between predicted output(outputs[0]) and the ground truth masks(targets)
            loss = criterion(outputs[0], targets.squeeze(1).long())
            
        #Scales the loss to safely compute gradients in mixed precision
        scaler.scale(loss).backward()
        #Applies the optimizer step using the scaled gradients.
        scaler.step(optimizer)
        #Updates the scaler for the next iteration (adapts the scale factor automatically)
        scaler.update()

        #Adds the batch loss to the total loss to compute the average later.
        running_loss += loss.item()

        #Increments the count of correctly classified pixels and total pixels to compute metrics
        _, predicted = torch.max(outputs[0], 1)
        correct += (predicted == targets.squeeze(1)).sum().item()
        total += torch.numel(targets.squeeze(1))

        #Frees memory by deleting unused variables
        del inputs, targets, outputs, predicted, loss
        gc.collect()
        
    # Compute Pixel Accuracy -> Even if mIoU is the correct metric for evaluation, we also calculate pixel accuracy during training to have 
    #an additional insight into the system’s performance.
    acc = 100. * correct / total
    print(f'Train Epoch {epoch+1} - Loss: {running_loss / len(train_loader):.4f} - Acc: {acc:.2f}%')

#-------------------------
#       VALIDATION 
#-------------------------

def validate(model, val_loader, criterion, device, num_classes=19):
    model.eval() #Sets the model to evaluation mode
    #initialization of the values
    val_loss, correct, total = 0, 0, 0
    miou_total = 0
    count = 0

    #Disables gradiend computation since we don’t want to update model weights.
    with torch.no_grad():
        #Iterates over batches from the validation dataset
        for inputs, targets in val_loader:
            #Moves the images and masks to the GPU 
            inputs, targets = inputs.to(device), targets.to(device)

            #Computes the loss between predicted output(outputs) and the ground truth masks(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze(1).long())
            #Adds the batch loss to the total loss to compute the average later.
            val_loss += loss.item()
            #Increments the count of correctly classified pixels and total pixels to compute metrics
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets.squeeze(1)).sum().item()
            total += torch.numel(targets.squeeze(1))

            #Compute mIou
            miou_batch = compute_miou(predicted, targets.squeeze(1), num_classes=num_classes)
            miou_total += miou_batch
            count += 1

            #Frees memory by deleting unused variables
            del inputs, targets, outputs, predicted, loss
            gc.collect()

    # Compute Pixel Accuracy -> Even if mIoU is the correct metric for evaluation, we also calculate pixel accuracy during training to have 
    #an additional insight into the system’s performance.
    acc = 100. * correct / total
    
    #After all batches are processed, the mean over all batches is computed:
    mean_iou = 100. * (miou_total / count) if count > 0 else 0
    print(f'Validation - Loss: {val_loss / len(val_loader):.4f} - Acc: {acc:.2f}% - mIoU: {mean_iou:.2f}%')
    return acc, mean_iou


#Function defined to find the correc folder of images/labels/train ecc in the Cityscapes folde
def find_folder(start_path, folder_name):
    for root, dirs, _ in os.walk(start_path):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None

#-------------------------
#       MAIN 
#-------------------------

if __name__ == "__main__":
    print(">>> Training...")

    #Find the path containing images and ground truth masks inside the extracted dataset directory.
    zip_path = 'cityscapes_dataset.zip'
    base_extract_path = './Cityscapes'
    images_dir = find_folder(base_extract_path, 'images')
    masks_dir = find_folder(base_extract_path, 'gtFine')
    if not images_dir or not masks_dir:
        raise RuntimeError("'images' o 'gtFine' non trovati dopo l’estrazione!")

    #Paths for train and validation sets
    train_images_dir = os.path.join(images_dir, 'train')
    val_images_dir = os.path.join(images_dir, 'val')
    train_masks_dir = os.path.join(masks_dir, 'train')
    val_masks_dir = os.path.join(masks_dir, 'val')

    #to found the base folder that contains both the images and masks folders as subfolders.
    base_path = os.path.commonpath([images_dir, masks_dir])

    #These functions generate CSV files that list image and mask pairs for training and validation.
    train_csv = 'train_annotations.csv'
    val_csv = 'val_annotations.csv'
    cityscapes.create_cityscapes_csv(train_images_dir, train_masks_dir, train_csv, base_path)
    cityscapes.create_cityscapes_csv(val_images_dir, val_masks_dir, val_csv, base_path)

    #Create dataset objects
    train_dataset = cityscapes.CityScapes(
        annotations_file=train_csv,
        root_dir=base_path,
        transform=cityscapes.transform['image'],
        target_transform=cityscapes.transform['mask']
    )
    val_dataset = cityscapes.CityScapes(
        annotations_file=val_csv,
        root_dir=base_path,
        transform=cityscapes.transform['image'],
        target_transform=cityscapes.transform['mask']
    )

    #Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Checkpoint folder creation
    os.makedirs('checkpoints', exist_ok=True)

    #Hyperparameters definition
    num_epochs = 50
    learning_rates = [0.000025, 0.001]
    batch_sizes = [1, 4, 8]
    #Best configuration lr=0.001 and bs=4
 
    #Loop over different hyperparamters configuration
    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\n>>> Training with lr={lr} and batch_size={bs}")
            
            #DataLoader for train and validation
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                      num_workers=2, pin_memory=True)  
            val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False,
                                    num_workers=2, pin_memory=True)

            #Initialize DeepLabV2 segmentation model for 19 classes
            model = get_deeplab_v2(num_classes=19).to(device)
            #Use cross-entropy loss without weight
            criterion = nn.CrossEntropyLoss(ignore_index=255)
            #Use SGD optimizer with momentum and weight decay.
            optimizer = optim.SGD(model.optim_parameters(lr = lr), lr=lr, momentum=0.9, weight_decay=5e-4)

            #Initialization of best values
            best_lr=0
            best_bs=0
            best_acc = 0
            best_miou = 0

            #This loop iterates over all epoches: for each epoch, it performs training 
            #and validation.
            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                train(epoch, model, train_loader, criterion, optimizer, device)
                val_acc,val_miou = validate(model, val_loader, criterion, device)
                
                #Define a periodic (each 10 epochs) checkipoints, where the models is saved 
                if epoch % 10 == 0 or epoch == num_epochs - 1:
                    torch.save(model.state_dict(), f'checkpoints/checkpoint_epoch_{epoch}_lr{lr}_bs{bs}.pth')
                    print(f"checkpoint epoch {epoch}")
                    
                #Checking the current mIou values and tracks the best mIous found
                if val_miou > best_miou:
                    best_miou = val_miou
                    #Saved the configuration of the best model
                    best_lr=lr
                    best_bs=bs
                    torch.save(model.state_dict(), f'checkpoints/best_model_lr{lr}_bs{bs}.pth')
                    print(f"Best model -> mIoU: {best_miou:.2f}% (Acc: {val_acc:.2f}%)")
              

            print(f"Training completed with lr={best_lr}, bs={best_bs} | Best mIou: {best_miou:.2f}%")
