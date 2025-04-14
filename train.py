import torch
import datasets.cityscapes 
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from models.deeplabv2.deeplabv2 import ResNetMulti

# Train the Model
def train(epoch, model, train_loader, criterion, optimizer):
    # Set the Model on the Training Mode
    model.train()

    # running_loss -> Total Loss (error) across the Whole Epoch
    running_loss = 0.0

    # correct -> #Images Predicted CORRECTLY
    correct = 0

    # total -> #Images Seen
    total = 0

    # Loops over EACH Batch in Training Set
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        #enumerate(train_loader): Iterates through the batches provided by the dataloader and keeps track of the index (batch_idx) and of the batch itself
        #(inputs, targets): For each batch, the DataLoader returns the input data (images as tensor) and the corresponding labels

        # Move inputs (batch of Images) and targets (True Class Labels) to GPU -> to improve the efficiency of the computations
        inputs, targets = inputs.cuda(), targets.cuda()

        # Pass the inputs (batch of Images) to the Model to obtain the outputs -> Tensor of logits [B, 200]
        #logits are the raw values ​​produced by the model before being converted into probabilities
        #(typically through an activation function such as Softmax or Sigmoid).
        #ES: Tensor of logits [2, 3]= logits = torch.tensor([[2.5, 0.3, -1.2] , [0.1,-0.5, 3.2]])= outputs
        outputs = model(inputs)

        # Compute the Loss/Error between the Predictions (outputs) and the True Labels(targets) accordin to the
        #criterion (type of used loss) passed as input
        # loss -> Single Scalar Loss Value
        loss = criterion(outputs, targets)

        # Backpropagation: TO UPDATE THE WEIGHTS AND IMPROVE THE MODEL
        optimizer.zero_grad() # -> Reset the Gradients of Model Parameters computed at the previous step
        loss.backward() # -> Compute the Gradients of Loss with respect to EACH Weight & EACH Bias
        optimizer.step() # -> Update the Model's Parameters using the Gradients
                             #(ask to the optimizer of follows the loss to understand how improve weights)
        #optimizer is the way that we use to update weights that is passed as input: SGD with Momentum, SGD, ADAGRAD ecc passed as input.

        # Add the Scalar Loss Value (.item() is Needed) to the Total Loss
        running_loss += loss.item()

        # Get the Predicted Class (Class with the Max Score from outputs, can be seen as the class with the higher probability):
        # _, -> NO Save the Max Score
        # predicted -> Save ONLY the Predicted Class associated wit the Max Score
        _, predicted = outputs.max(1)

        # #Samples that were in this Batch:
        # - targets -> Tensor of shape: [B]
        # - .size(0) -> Return #Elements in targets
        total += targets.size(0)

        # Compare Predicted Labels (predicted) and True Labels (True Labels)
        # - .sum() -> Sum the 1s (Correct Predictions in the Batch)
        # - .item() -> Extract the Scalar Value
        correct += predicted.eq(targets).sum().item()

    # Compute the Average Loss per Batch
    train_loss = running_loss / len(train_loader)

    # Compute the Overall Accuracy for the Epoch
    train_accuracy = 100. * correct / total

    # DEBUG: to understand where and if the algorithm performs bad
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')





# Validate the Model
def validate(model, val_loader, criterion):
    # Set the Model on the Evaluation Mode
    model.eval()

    # val_loos -> Total Loss across ALL Validation Batches
    val_loss = 0

    # correct -> #Images Predicted CORRECTLY
    correct = 0

    # total -> #Images Seen
    total = 0

    # Disable Gradient Calculations (BECAUSE we're in Validation)
    with torch.no_grad():
        # Loops over EACH Batch in the Validation Set
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # Move inputs (batch of Images) and targets (True Class Labels) to GPU
            inputs, targets = inputs.cuda(), targets.cuda()

            # Pass the inputs (batch of Iamges) to the Model:
            # outputs -> Tensor of logits [B, 200]
            outputs = model(inputs)

            # Compute the Loss/Error
            # between the Predictions (outputs) and the True Labels(targets)
            # loss -> Single Scalar Loss Value
            loss = criterion(outputs, targets)

            # Add the Scalar Loss Value (.item() is Needed) to the Total Loss
            val_loss += loss.item()

            # Get the Predicted Class (Class with the Max Score from outputs):
            # _, -> NO Save the Max Score
            # predicted -> Save ONLY the Predicted Class associated wit the Max Score
            _, predicted = outputs.max(1)

            # #Samples that were in this Batch:
            # - targets -> Tensor of shape: [B]
            # - .size(0) -> Return #Elements in targets
            total += targets.size(0)

            # Compare Predicted Labels (predicted) and True Labels (True Labels)
            # - .sum() -> Sum the 1s (Correct Predictions in the Batch)
            # - .item() -> Extract the Scalar Value

            #.eq() let us to compare predicted and targets, with sum we sum how many times we have true
            correct += predicted.eq(targets).sum().item()

    # Compute the Average Loss per Batch
    val_loss = val_loss / len(val_loader)

    # Compute the Overall Accuracy for the Epoch
    val_accuracy = 100. * correct / total

    # DEBUG:
    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy
# Define the Hyperparameters

#From the indication we know:
#Report results for:
#○ LR: 0.1, 0.001, 0.0001
#○ Batch Size: 16, 32, 64

learning_rates = [0.01, 0.001,0.0001]  #Controls how much the model updates its parameters at each step:
                                       # 0.01 means that model quickly update the weights (changing its values quickly)
batch_sizes = [16,32,64]               #number of samples that the model "check" at each iteration and base on them update the weights
                                       #to higher means to higher complexity, so low efficiency


# Define the Transformations for the Dataset:
#Why performs TRASFORMATIONS?
# We use transforms to perform some manipulation of the data and make it suitable for training,
#a sort of data augmentation and data normalization to have a suitable dataset.

'''
# Trainining Transformations (Improve Generalization):
'train' : transforms.Compose([
    transforms.Resize((227, 227)),  # Resize the Images to Match AlexNet input
    transforms.RandomHorizontalFlip(), #Randomly flips an image along the horizontal axis with probability p=0.5
    transforms.ToTensor(),  #Converts image into a numpy array
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Normalize a tensor image with mean and standard deviation
                                                                    #adjusts the range of pixel values in an image to a standard range,
                                                                    #to helps neural networks process images more effectively. The values are related to
                                                                    #ImageNet, here we use Tiny ImageNet that is a small version of ImageNet.


]),

# Validation Transformations (Evaluate Performance CONSISTENTLY):
'val' : transforms.Compose([
    transforms.Resize((227, 227)),  # Resize the Images to Match AlexNet input
    transforms.ToTensor(), #Converts image into a numpy array
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ##Normalize a tensor image with mean and standard deviation
])
}
'''


# Load the Tiny ImageNet Dataset

dataset_full = datasets.cityscapes.CityScapes()

train_dataset = dataset_full.split('train')
val_dataset = dataset_full.split('val')
#datasets.ImageFolder is a class of PyTorch to load and collect images, using trasform we apply the previous transformation

# Define #Epochs-> An epoch represents a complete cycle through the entire training dataset.
                  #In other words, an epoch is completed when every sample in the dataset has been used once to update the model's weights.
                  #The loop is:   for epoch in range(num_epochs) and an epoch is obtained by train(epoch, resnet, train_loader, criterion, optimizer)
                  #where epoch passed as parameter is a simple number

                  #An iteration refers to a single update of the model weights. It occurs within an epoch and represents the process
                  #of training on a batch of data.
                  #The loop is:  for batch_idx, (inputs, targets) in enumerate(train_loader)
                  # at each iteration, the model has seen a batch, at the end of the loop it compute an epoch

                  #ES: Epoch 1:
                  #             |-- Iterazione 1: Batch 1 (Immagini 1-100)
                  #             |-- Iterazione 2: Batch 2 (Immagini 101-200)
num_epochs = 50

# Loop over the Hyperparameters
for lr in learning_rates: #learning_rates = [0.01, 0.001,0.0001]
    for batch_size in batch_sizes: #batch_sizes = [16,32,64]
        # Data Loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the AlexNet
        resnet = ResNetMulti(num_classes=19).cuda()

        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss() #CrossEntropy is the type of used Loss
        optimizer = optim.SGD(resnet.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        #The method choosen to update the weight is Stochastic Gradient Descent (SGD)
        # -resnet.parameters() say that it must be applied on the weights of AlexNet, lr, momentum and weight_decay define the SGD

        best_acc = 0

        # Train & Validate the AlexNet
        for epoch in range(num_epochs):
            # Training Loop
            train(epoch, resnet, train_loader, criterion, optimizer) #an epoch, the model has seen the entire dataset

            # Validation Loop
            val_accuracy = validate(resnet, val_loader, criterion)

            # Evaluate on the Validation Set and Print the Results
            best_acc = max(best_acc, val_accuracy)

        print(f'Best Validation Accuracy: {best_acc:.2f}%')

        # IF: we want to Keep the Model State
        # THEN: Save the Model State
        # torch.save(resnet.state_dict(), f'alexnet_lr{lr}_bs{batch_size}.pth')
