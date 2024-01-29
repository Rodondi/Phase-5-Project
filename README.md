# Pneumonia Detection using Chest X-Ray Images
# Authors..

Allan Ngeiywa

Cindy King'ori

Eunice Nduati

Lucy Munge

Muthoni Kahuko

Rodgers Bob

Samuel Lumumba


![image](https://github.com/Rodondi/Phase-5-Project/assets/133041685/8e3d4c95-dd36-4600-bfbb-626bcff1dcab)


![image](https://github.com/Rodondi/Phase-5-Project/assets/133041685/d6f4c432-81f3-48f4-b173-2fd538e4132d)

# Problem Statement:

Pneumonia is form of a respiratory infection that affects the lungs. In these acute respiratory diseases, human lungs which are made up of small sacs called alveoli which in air in normal and healthy people but in pneumonia these alveoli get filled with fluid or "pus” one of the major step of phenomena detection and treatment is getting the chest X-ray of the (CXR). Chest X-ray is a major tool in treating pneumonia, as well as many decisions taken by doctor are dependent on the chest X-ray. 
Our project is about detection of Pneumonia by chest X-ray using Convolutional Neural Network.
Early and reliable detection of pneumonia can contribute to more timely medical intervention and improved patient outcomes.

# Main Objectives

- Research, design, and implement advanced algorithms for the accurate detection of pneumonia in chest X-ray images.
- Construct a robust binary classifier capable of distinguishing between normal and pneumonia cases in chest X-ray images.
- Integrate the developed pneumonia detection algorithms into an automated diagnostic tool for chest X-ray images.
- Improve the efficiency and precision of pneumonia diagnosis by deploying the automated diagnostic tool.

# Specific Objective

The goal of this project is to develop an automated system for detecting and classifying pneumonia in medical images. Design and implement a robust deep learning algorithm, for detecting and classifying pneumonia in chest X-ray images.

# Data Understanding

The dataset, sourced from Kaggle and accessible at https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data , is organized into three main folders: train, test, and val. Each folder includes subfolders representing two distinct image categories: Pneumonia and Normal. The dataset comprises a total of 5,856 chest X-ray images in JPEG format. These images, employing the anterior-posterior technique, originate from retrospective cohorts of pediatric patients aged one to five years at the Guangzhou Women and Children’s Medical Center in Guangzhou, China. The inclusion of these chest X-ray images in the dataset was part of routine clinical care for pediatric patients.

Ensuring dataset quality, an initial screening process eliminated low-quality or unreadable scans to minimize errors. Two expert physicians then meticulously graded the diagnoses associated with the images, deeming them suitable for training the AI system only after this rigorous evaluation. To further mitigate potential grading errors, an additional layer of scrutiny was applied to the evaluation set. This involved examination by a third expert, providing an extra level of assurance to the accuracy of the diagnoses. This comprehensive approach to quality control and grading establishes a robust foundation for the analysis of chest X-ray images, enhancing the reliability of the AI system trained on this dataset.
![image](https://github.com/Rodondi/Phase-5-Project/assets/133041685/81c87b51-4329-4350-9210-93be1e232590)

# Data Preparation

![image](https://github.com/Rodondi/Phase-5-Project/assets/133041685/7e4e56d5-327f-45f0-8efc-9125fcee21ea)
Number of files in train folder: 5216
Number of files in val folder: 16
Number of files in test folder: 624

Number of files in each subfolder of train folder:
.: 0
NORMAL: 1341
PNEUMONIA: 3875

Number of files in each subfolder of val folder:
.: 0
NORMAL: 8
PNEUMONIA: 8

Number of files in each subfolder of test folder:
.: 0
NORMAL: 234
PNEUMONIA: 390

# Data Cleaning
Detecting whether an image is blurred based on the variance of the Laplacian. This is a common technique to identify blurred images in image processing applications.

Number of low-quality images in train folder: 1671
Number of low-quality images in val folder: 1
Number of low-quality images in test folder: 263

The data output indicates a notable presence of low-quality images. Subsequently, we undertake a detailed examination of specific quality issues to enhance resolution.

# Pre-processing:
Data augmentation techniques serve the purpose of artificially increasing the diversity of a dataset by applying various transformations to the existing data. In the context of image data, augmentation involves creating new images by making slight modifications to the original ones. The primary purposes of augmentation techniques are: increased diversity, improved robustness, reduced over-fitting, better generalisations and enhanced training efficiency.
![image](https://github.com/Rodondi/Phase-5-Project/assets/133041685/eabdeed8-d50d-4611-a842-15b234ce6de5)

# EDA
Class Distribution Visualization
![image](https://github.com/Rodondi/Phase-5-Project/assets/133041685/5fd97be1-ab3c-468c-a57a-46c3f63b3cdb)
Image Dimensions

Image Quality Assessment
<img width="475" alt="image" src="https://github.com/Rodondi/Phase-5-Project/assets/133041685/3576c669-b6a8-4715-9856-6de58cbaa50a">
![image](https://github.com/Rodondi/Phase-5-Project/assets/133041685/7c72f929-53f0-4e1e-b447-3194baf78b76)

Outlier Detection

Image Dimensions Outliers
![image](https://github.com/Rodondi/Phase-5-Project/assets/133041685/05848f54-9258-4533-9d8e-4dc28d0953dd)
Pixel Intensity Outliers:
![image](https://github.com/Rodondi/Phase-5-Project/assets/133041685/3d13ecda-c1c1-451e-aade-4526fd49c733)

# Modelling

Creating a Weighted Loss Function To Address Class Imbalance

We address the issue of class imbalance in a binary image classification task using TensorFlow and Keras. The goal is to create a weighted loss function that assigns different weights to classes based on their frequencies in the training data.

Found 4173 images belonging to 2 classes
Weight for class 0: 0.74
Weight for class 1: 0.26

Convolutional Neural Network

Total params: 27,841
Trainable params: 27,841
Non-trainable params: 0

![image](https://github.com/Rodondi/Phase-5-Project/assets/133041685/b5c8b9cc-216d-4508-be74-8511e7251b24)

<img width="511" alt="image" src="https://github.com/Rodondi/Phase-5-Project/assets/133041685/1c3d37b4-0379-48c8-8be7-562fb4794b2b">


DenseNet

Total params: 11,233,602
Trainable params: 11,149,442
Non-trainable params: 84,160

<img width="368" alt="image" src="https://github.com/Rodondi/Phase-5-Project/assets/133041685/bfde2e66-5511-4159-82a7-152976f2bc6b">


InceptionNet

Found 4173 images belonging to 2 classes.
Weight for class 0: 0.74
Weight for class 1: 0.26

Total params: 22,328,609
Trainable params: 22,293,665
Non-trainable params: 34,944

# Confusion Matrix:

![image](https://github.com/Rodondi/Phase-5-Project/assets/133041685/3246f00c-67c0-49ab-89b3-f2bf730f7bc7)
![image](https://github.com/Rodondi/Phase-5-Project/assets/133041685/e19a0483-0b97-453d-bb46-e95a28b2ca5e)
![image](https://github.com/Rodondi/Phase-5-Project/assets/133041685/f8eaf491-8367-406f-bcc9-4da630fbe4ef)

# Observations


# Conclusions

Reliable recognition of infections in the lung is a key step in the diagnosis of Pneumonia disease. X-ray imaging examination of Chest is usually performed by trained human examiners or doctors, making the process time-consuming and hard to standardize. 

This Project is a Pneumonia detection model using the DenseNet Model and Pneumonia Chest X-ray dataset. This data was collected from the various patients and clinically examined and categorized by human examiners. The proposed DenseNet Model was trained on by using 1000 training epochs and TensorFlow framework. The training process of the model uses 5216 chest X-ray Images and the testing process uses 624 images. 
The performance of the proposed model used, evaluated thus using different metrics such as Classification accuracy, Sensitivity, Specificity and the F1 score. The Classification accuracy of the proposed model achieved the average accuracy of 87.5 percentage in unseen chest X-ray images. Also, this accuracy was greater than the existing transfer learning approaches such as CNN and InceptionNet. The proposed DenseNet Model was found most suitable to detect Pneumonia infection from Chest X-ray images.

In the future, different lung disease classes will include this model to detect various lung diseases using the chest X-ray images. Also, the performance of the proposed Deep CNN model can be improved with more number of layers and parameters. This will allow clinicians to recognize lung diseases from chest X-ray images with lower prevalence at an earlier stage of the disease.

# Recommendation

Potential areas for improvement include fine-tuning, additional data collection, and exploring alternative architectures to enhance the model's performance.
