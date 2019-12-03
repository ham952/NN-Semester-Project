# OBJECT CLASSIFICATION USING NEURAL NETWORKS
This repo contains code used in the implementation of term project of Neural Network Course. Please go through requirements.txt for code execution guidance. Feel free to use , dont forget to acknowledge.
 
 ## Objective 
The main aim of this project is to implement and analyze the problem of object classification using neural networks. Firstly, we have implementing deep neural networks and evaluated the results of the same problem. The problem is then investigated by implementing Convolutional Neural Networks (CNN). Finally the problem is studied under the more recent deep CNN architecture of ResNet50.
   

## Brief Description of Proposed Research
Object classification is one of the classical problem of computer vision. The problem has been thoroughly investigated using classical machine learning as well as by using neural networks. This project uses neural networks to study the same problem. We have first implemented a generic architecture of neural networks using basic python without using any machine learning libraries. This architecture uses back-propagation to update its parameters through gradient descent. On the top level,  user can define number of layers and number of neurons of each layer. As a result, the desired architecture is generated. The same architecture was tested against different random variations in number of layers as well as by varying number of neurons in each layer. The Implementation on the basic level helped us in a better understanding of the architecture as well as gave user more control over the basic parameters. This architecture is tested and analyzed over a custom “cats/no cats” dataset. 

We have then analyzed object classification under Convolutional Neural Network (CNN). CNN offer advantage over traditional methods in simultaneous feature extraction and classification as well as a strong generalization ability. Instead of using raw pixel as input to our network, CNN extracts desired features out of the target image to solve the problem. Furthermore, CNN does not need to extract specific manual features from image for a particular task. They perform hierarchical abstraction processing on the original image to generate the required results. Parameters to be trained by CNN are far less in number as compared to, when using raw pixel values. But the results are comparable and even better than using raw pixels. This motivated us to compare deep neural network architecture against convolutional neural network. To compare the two networks on a common dataset, we first implemented the same deep neural network architecture using Keras framework. This is followed by a custom convolutional neural network architecture in Keras. The “accuracy” and “number of trainable parameters” and “execution time” to reach given accuracy, taken by the two networks are compared. We have evaluated the two networks are over “signs dataset” from Kaggle. It is pertinent to mention that the binary classification problem of the last step has now been changed to a multi-class classification problem by using “signs dataset”.

The last phase of this project involves the study of deep neural networks architectures available in literature. A brief overview of LeNet-5, AlexNet and VGG16 is given. This is followed by implementation of ResNet50 in Keras. This architecture was tested on the same dataset and the results were compared with the earlier developed architectures of deep neural networks and convolutional neural networks

## Results

### 1. DNN in Pyhton on cats/no cats dataset
|  Layer dimensions | Layers Count | Train Accuracy  |Test Accuracy|
|     :---:      |     :---:      |     :---:      |     :---:      |
| [12288,7,1] | 2 | 1.00| 0.74 |
|   [12288,28,6,1] | 3| 1.00| 0.84 |
|   [12288,20,7,5,1] | 4| 0.9952| 0.80 |
|   [12288,28,18,9,4,1] | 5| 1.00| 0.74 |
|   [12288,60,35,20,18,7,4,1] | 7| 1.00| 0.58 |

### 2. DNN Vs CNN in Keras on Signs Dataset

|  Parameter | DNN in Keras | CNN in Keras  |
|     :---      |     :---:      |     :---:      |    
| Trainable Parameters | 307,615 (~0.3m) | 10,214| 
|   Test  Accuracy |  0.7833 (max 0.825) | 0.82499|
|   Train Accuracy |   0.9898 |  0.9741|
|   Train Loss |  0.0474 |  0.1383|
|   Execution Time |   9.8988  mins |  5.33754  mins|
|   Epochs |   1000 |  120|

### 3. ResNet50 Vs custom CNN on Signs Dataset

|  Parameter | CNN in Keras | Resnet50  |"|
|     :---      |     :---:      |     :---:      |     :---:      |
| Epochs | 120 | 2| 5 |
|   Test  Accuracy |  0.824999 | 0.7833| 0.8916 |
|   Time  taken |   5.33754  mins |  3.9617  mins| 6.853  mins  |
 

   
 


  



  
 



