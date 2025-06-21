# Improving Domain Generalization in Real-Time Semantic Segmentation through Rare-Class Tailored Techniques

This document is intended to guide users through the repository structure and provide clear instructions on how to navigate and utilize the project components effectively.

---

## Project Steps and Corresponding Scripts

### **Step 2 – Testing Semantic Segmentation Networks**
- **2.a**: Use `trainDeeplab.py` to train the DeepLabV2 model.
- **2.b**: Use `trainBisenet.py` to train the BiSeNet model.
- Evaluation:
  - Use `evaluationDeepLab.py` for evaluating DeepLabV2.
  - Use `evaluationBiseNet.py` for evaluating BiSeNet.
- Both models are trained and evaluated on the **Cityscapes** dataset defined in `cityscapes.py`.

---

### **Step 3 – Domain Shift Evaluation**
- Use `trainDiffAug.py` to train BiSeNet on **GTA5** without augmentation (or with custom augmentation combinations).
- Augmentations are defined in `augmentation.py`.
- The script automatically calls `preprocess_mask.py` to generate semantic masks from GTA5 labels.
- Evaluation can be performed using `trainBisenet.py` with a validation split from **Cityscapes**.

---

### **Step 4 – Domain Adaptation**
- Use `trainDomainAdapt.py` for training with **domain adaptation techniques**.
  - This script uses the **best augmentation configuration**, defined in `augmentation.py`.
  - Training is conducted on **GTA5**, specified in `gta5.py`.
  - `preprocess_mask.py` is used for preparing masks.
  - `focalLoss.py` contains the implementation of **Focal Loss** for rare-class handling.
  - `discriminator.py` implements the **discriminator network** used for adversarial training.
- Evaluation is done using the standard evaluation script: `trainBisenet.py`.

---
## Structure

SemSeg_MLDL25/

│

├── datasets/ 

│   ├── cityscapes.py               # Cityscapes dataset loader

│   └── gta5.py                     # GTA5 dataset loader

│

├── models/                         # Network architectures

│   ├── bisenet/

│   │   ├── build_bisenet.py        # Constructs the BiSeNet model

│   │   └── build_contextpath.py    # Defines context path/backbone for BiSeNet

│   │

│   ├── deeplabv2/

│   │   └── deeplabv2.py            # DeepLabV2 model definition

│

├── discriminator.py                # Discriminator model for adversarial training

│

├── trainDeeplab.py                 # Training script for DeepLabV2

├── evaluationDeepLab.py            # Evaluation script for DeepLabV2

│

├── trainBisenet.py                 # Training script for BiSeNet

├── evaluationBiseNet.py            # Evaluation script for BiSe

---
## Extension

Our extension involves several techniques aimed at improving the recognition of rare classes and is fully integrated into the project. In particular, it introduces class mix within the augmentation process, as well as the use of specific entropy measures both during augmentation and domain adaptation phases. Additionally, a targeted pretraining approach is implemented to further enhance the model’s ability to accurately recognize less frequent classes

---
## Prerequisites

The `fvcore` library is required to the evaluation step. You can install it using pip:

```bash
!pip install -U fvcore


.
