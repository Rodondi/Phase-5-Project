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

Research, design, and implement advanced algorithms for the accurate detection of pneumonia in chest X-ray images.
Construct a robust binary classifier capable of distinguishing between normal and pneumonia cases in chest X-ray images.
Integrate the developed pneumonia detection algorithms into an automated diagnostic tool for chest X-ray images.
Improve the efficiency and precision of pneumonia diagnosis by deploying the automated diagnostic tool.

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

# Modelling

 Convolutional Neural Network

 DenseNet

 InceptionNet

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
