# Handwritten-Digit-Classification-using-KNN
This repository introduces to my project "Handwritten-Digit-Classification" using MNIST Data-set . This project was implemented and executed by applying KNN algorithm with recognition accuracy of around 91-93 % . The desired results have been obtained by training the machine first using the mnist_train data-set and later testing the obtained results using mnist_test data-set , to recognise the handwritten digit.  

# Introduction
The aim of this project is to implement a classification algorithm to recognize handwritten digits (0‐ 9). It has been shown in pattern recognition that no single classifier performs the best for all pattern classification problems consistently. Hence, the scope of the project also included the elementary study the different classifiers and combination methods, and evaluate the caveats around their performance in this particular problem of handwritten digit recognition. This report presents our implementation of the Principal Component Analysis (PCA) combined with 1‐Nearest Neighbor to recognize the numeral digits, and discusses the other different classification patterns. I was able to achieve an accuracy rate of 92.8%.

# Overview of the Project
Hand writing recognition of characters has been around since the 1980s.The task of handwritten digit recognition, using a classifier, has great importance and use such as – online handwriting recognition on computer tablets, recognize zip codes on mail for postal mail sorting, processing bank check amounts, numeric entries in forms filled up by hand (for example ‐ tax forms) and so on. There are different challenges faced while attempting to solve this problem. The handwritten digits are not always of the same size, thickness, or orientation and position relative to the margins. My goal was to implement a pattern classification method to recognize the handwritten digits provided in the MINIST data set of images of hand written digits (0‐9). The data set used for our application is composed of 300 training images and 300 testing images, and is a subset of the MNIST data set [1] (originally composed of 60,000 training images and 10,000 testing images). Each image is a 28 x 28 grayscale (0‐255) labeled representation of an individual digit.   The general problem we predicted we would face in this digit classification problem was the similarity between the digits like 1 and 7, 5 and 6, 3 and 8, 9 and 8 etc. Also people write the same digit in many different ways ‐ the digit ‘1’ is written as ‘1’, ‘1’, ‘1’ or ‘1’. Similarly 7 may be written as 7, 7, or 7. Finally the uniqueness and variety in the handwriting of different individuals also influences the formation and appearance of the digits.  


![image](https://user-images.githubusercontent.com/46643368/68531060-6374c080-0334-11ea-85ca-a81156101af1.png)

Fig. 1 Sample digits used for training the classifier  

# Literature Survey
Hand Written Character Recognition using Star-Layered Histogram Features 
Stephen Karungaru, Kenji Terada and Minoru Fukumi 
In this method, a character recognition method using features extracted from a  star layered histogram is presented and trained using neural networks. After  several image preprocessing steps, the character region is extracted. Its contour is  then used to determine the center of gravity (COG). This CoG point is used as the  origin to create a histogram using equally spaced lines extending from the CoG to  the contour. 
The first point the line touches the character represents the first layer of the  histogram. If the line extension has not reached the region boundary, the next hit  represents the second layer of the histogram. This process is repeated until the  line touches the boundary of the character’s region. After normalization, these  features are used to train a neural network. 

# Problem Statement
Given a set of greyscale isolated numerical images taken from 
MNIST database. 

The objectives are:- 

	To recognize handwritten digits correctly. 

	To improve the accuracy of detection. 

	To develop a method which is independent of digit size and  writer style/ink independent. 

# Scope of Study
•	Postal mail sorting 
•	Courtesy amounts on cheques 
•	Formation of data entry etc. 
# System Analysis
The first job of the group of students is to divide the different tasks between all the members of the group and organize the communication to be able to integrate all the parts before the evaluation of the project. 
System Design

•	KNN algorithm
K-Nearest Neighbors (or KNN) is a simple classification algorithm that is surprisingly effective. However, to work well, it requires a training dataset: a set of data points where each point is labelled (i.e., where it has already been correctly classified). If we set K to 1 (i.e., if we use a 1-NN algorithm), then we can classify a new data point by looking at all the points in the training data set, and choosing the label of the point that is nearest to the new point. If we use higher values of K, then we look at the K nearest points, and choose the most frequent label amongst those points. 

However, in order to apply the k-Nearest Neighbor classifier, we first need to select a distance metric or a similarity function. We briefly discussed the Euclidean distance (often called the L2-distance) in our project:

![image](https://user-images.githubusercontent.com/46643368/68531191-b733d980-0335-11ea-8348-72e12e358259.png)

•MNIST Dataset
The MNIST handwritten digit classification problem is a standard dataset used in computer vision and deep learning. Although, the dataset is effectively solved, it can be used as the basis for learning and practicing how to develop, evaluate, and use convolutional deep learning neural networks for image classification from scratch. This includes how to develop a robust test harness for estimating the performance of the model, how to explore improvements to the model, and how to save the model and later load it to make predictions on new data.
 
 # System Overview 
Our approach to solve this problem of handwritten numeral recognition can be broadly divided into three blocks: 
i) Pre‐Processing/Digitization ii) Feature Extraction using PCA iii) Classification using 1‐Nearest Neighbor algorithm; The block diagram for the system is shown below (Fig. 2): 
 

![image](https://user-images.githubusercontent.com/46643368/68530843-345d4f80-0332-11ea-9c14-c75551016a3d.png)

Fig.2. System Diagram of the implemented pattern classifier


# Testing
•	The overall classification design of the MNIST digit database is shown in following algorithm. Algorithm: Classification of Digits Input: Isolated Numeral images from MNIST Database Output: Recognition of the Numerals Method: Structural features and KNN classifier.

• Step 1: Convert the gray level image into Binary image

•	Step 2: Preprocessing the Binary Image

•	Step 3: Convert the Binary Image into a single Dimensional Array of [1,n]

•	Step 4: Keep the label of each Array along with it.

•	Step 5: Feed the classifier with the train_data set.

•	Step 6: Repeat the steps from 1 to 5 for all images in the Sample and Test Database. 

•	Step 7: Estimate the minimum distance between feature vector and vector stored in the library by using Euclidian distances.

•	Step 8: Feed the classifier with test_data set.

•	Step 9: Classify the input images into appropriate class label using minimum distance K-nearest neighbor classifier.

•	Step10: End. 

# How to Run ?

The project was executed using Anaconda software, and this source code file was tested on Jupyter notebook , so do ensure that you must have Anaconda software installed on your systems.
After installation of Anaconda software , follow the steps mentioned below:-

Step-1 : Upload the source code file in the local directory of Jupyter notebook, by just clicking on upload button in the Home directory of Jupyter notebook.

Step-2 : Similarly, upload the train_data and test_data dataset files by unzipping the files on your systems.

Step-3 : Simply, execute the source code in the Jupyter notebook cells.  

# Implementation
In short, the problem of Handwritten digits Classification is solve by  k-nearest-neighbors algorithm using MNIST data set. 
The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset.
It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.
The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.
It is a widely used and deeply understood dataset and, for the most part, is “solved.” Top-performing models are deep learning convolutional neural networks that achieve a classification accuracy of above 99%, with an error rate between 0.4 %and 0.2% on the hold out test dataset.
We obtain our data from the MNIST Data Set, and with a few minor modifications, A single line of the data file represents a handwritten digit and its label. The digit is a 256-element vector obtained by flattening a 16×16 binary-valued image in row-major order; the label is an integer representing the number in the picture. The data file contains 1593 instances with about 160 instances per digit.
After reading in the data appropriately, we randomly split the data set into two pieces, train on one piece, and test on the other. The following function does this, returning the success rate of the classification algorithm on the testing piece.
A run with   gives a surprisingly good 89% success rate. Varying  , we see this is about as good as it gets without any modifications to the KNN algorithm
Of course, there are many improvements we could make to this naive algorithm. But considering that it utilizes no domain knowledge and doesn’t manipulate the input data in any way, it’s not too shabby.
As a side note, it would be fun to get some tablet software and have it use this method to recognize numbers as one writes it. Alas, we have little time for these sorts of applications.

# Screenshots

![Screenshot (36)](https://user-images.githubusercontent.com/46643368/68530344-75069a00-032d-11ea-908f-7ffe3ea69f2a.png)

![Screenshot (37)](https://user-images.githubusercontent.com/46643368/68530390-bf881680-032d-11ea-9f14-cd9fd6603eee.png)


![Screenshot (38)](https://user-images.githubusercontent.com/46643368/68530394-c878e800-032d-11ea-8472-be42ca2493a8.png)


![Screenshot (40)](https://user-images.githubusercontent.com/46643368/68530400-cd3d9c00-032d-11ea-9985-63dd9017e2e2.png)

![Screenshot (41)](https://user-images.githubusercontent.com/46643368/68530403-d464aa00-032d-11ea-8046-a9c4536762bb.png)


![Screenshot (42)](https://user-images.githubusercontent.com/46643368/68530407-d9295e00-032d-11ea-9602-add99102a620.png)

![Screenshot (43)](https://user-images.githubusercontent.com/46643368/68530413-df1f3f00-032d-11ea-8945-8849cd37474e.png)


# Output 

![Screenshot (44)](https://user-images.githubusercontent.com/46643368/68530429-18f04580-032e-11ea-8212-f0379b2d8b3b.png)

![Screenshot (45)](https://user-images.githubusercontent.com/46643368/68530430-1b529f80-032e-11ea-8c9b-20eafcd32860.png)

![Screenshot (46)](https://user-images.githubusercontent.com/46643368/68530433-1f7ebd00-032e-11ea-90f6-6a03a9aca2ef.png)

![Screenshot (48)](https://user-images.githubusercontent.com/46643368/68530438-23124400-032e-11ea-81f4-e68a06ab0257.png)

![Screenshot (49)](https://user-images.githubusercontent.com/46643368/68530441-29a0bb80-032e-11ea-95dc-3a1402b4edab.png)

# Summary
In this project, we learnt about the most simple machine learning classifier — the k-Nearest Neighbor classifier, or simply k-NN for short. The k-NN algorithm classifies unknown data points by comparing the unknown data point to each data point in the training set. This comparison is done using a distance function or similarity metric. Then, from the k most similar examples in the training set, we accumulate the number of “votes” for each label. The category with the highest number of votes “wins” and is chosen as the overall classification.

While simple and intuitive, and though it can even obtain very good accuracy in certain situations, the k-NN algorithm has a number of drawbacks. The first is that it doesn’t actually “learn” anything — if the algorithm makes a mistake, it has no way to “correct” and “improve” itself for later classifications. Secondly, without specialized data structures, the k-NN algorithm scales linearly with the number of data points, making it a questionable choice for large datasets.

# Conclusion
The proposed project shows the whole process of supervised classification from the data acquisition to the design of a classification system and its evaluation. The project is usually extremely successful with the students. Most of them are very proud of the performance of their system and most of the time, they implement side games such as Sudoku or letter games (such as scrabble) to show the performance of their system online. The project, besides giving practical applications to some machine learning techniques seen in class, gives them the opportunity to work as a group. As explained in the text, some changes can be made on the representations of the images and on the distances used to compare two elements in the database.
