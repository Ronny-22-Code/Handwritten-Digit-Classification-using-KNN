# Handwritten-Digit-Classification-using-KNN
This repository introduces to my project "Handwritten-Digit-Classification" using MNIST Data-set . This project was implemented and executed by applying KNN algorithm with recognition accuracy of around 91-93 % . The desired results have been obtained by training the machine first using the mnist_train data-set and later testing the obtained results using mnist_test data-set , to recognise the handwritten digits.  
We’re going to start this lesson by reviewing the simplest image classification algorithm: k-Nearest Neighbor (k-NN). This algorithm is so simple that it doesn’t do any actual “learning” — yet it is still heavily used in many computer vision algorithms. And as you’ll see, we’ll be able to utilize this algorithm to recognize handwritten digits from the popular MNIST dataset.

# Objectives:
By the end of the project , you will :-

Have an understanding of the k-Nearest Neighbor classifier.
Know how to apply the k-Nearest Neighbor classifier to image datasets.
Understand how the value of k impacts classifier performance.
Be able to recognize handwritten digits from (a sample of) the MNIST dataset.
The k-Nearest Neighbor Classifier
The k-Nearest Neighbor classifier is by far the most simple image classification algorithm. In fact, it’s so simple that it doesn’t actually “learn” anything! Instead, this algorithm simply relies on the distance between feature vectors, much like in building an image search engine — only this time, we have the labels associated with each image, so we can predict and return an actual category for the image.

Simply put, the k-NN algorithm classifies unknown data points by finding the most common class among the k closest examples. Each data point in the k closest data points casts a vote, and the category with the highest number of votes wins! Or in plain english: “Tell me who your neighbors are, and I’ll tell you who you are”.

In order for the k-NN algorithm to work, it makes the primary assumption that feature vectors that lie close together in an n-dimensional space have similar visual contents:


However, in order to apply the k-Nearest Neighbor classifier, we first need to select a distance metric or a similarity function. We briefly discussed the Euclidean distance (often called the L2-distance) in our lesson on color channel statistics:

In reality, you can use whichever distance metric/similarity function most suits your data (and gives you the best classification results). However, for the remainder of this lesson, we’ll be using the most popular distance metric: the Euclidean distance.

# k-NN in action

At this point, we understand the principles of the k-NN algorithm. We know that it relies on the distance between feature vectors to make a classification. And we know that it requires a distance metric/similarity function to compute these distances.

# But how do we actually make the classification?

We can keep performing this process for varying values of k, but no matter how large or small k becomes, the principle remains the same — the category with the largest number of votes in the k closest training points wins and is used as the label for the testing point.

# Recognizing handwritten digits using MNIST

 We’ll be using the k-Nearest Neighbor classifier to classify images from the MNIST dataset, which consists of handwritten digits. The MNIST dataset is one of the most well studied datasets in the computer vision and machine learning literature. In many cases, it’s a benchmark and a standard to which machine learning algorithms are ranked.

The goal of this dataset is to correctly classify the handwritten digits 0-9. Instead of utilizing the entire dataset (which consists of 60,000 training images and 10,000 testing images,) we’ll be using a small subset of the data provided by the scikit-learn library — this subset includes 1,797 digits, which we’ll split into training, validation, and testing sets, respectively.

Each image in the 1,797-digit dataset from scikit-learn is represented as a 64-dim raw pixel intensity feature vector. This means that each image is actually an 8 x 8 grayscale image, but scikit-learn “flattens” the image into a list.

All digits are placed on a black background with the foreground being shades of white and gray.

Our goal here is to train a k-NN classifier on the raw pixel intensities and then classify unknown digits.

To accomplish this goal, we’ll be using our five-step pipeline to train image classifiers:

Step 1 – Structuring our initial dataset: Our initial dataset consists of 60,000 digits representing the numbers 0-9. These images are grayscale, 8 x 8 images with digits appearing as white on a black background. These digits have also been heavily pre-processed, aligned, and centered, making our classification job slightly easier.
Step 2 – Splitting the dataset: We’ll be using three splits for our experiment. The first set is our training set, used to train our k-NN classifier. We’ll also use a validation set to find the best value for k. And we’ll finally evaluate our classifier using the testing set.
Step 3 – Extracting features: Instead of extracting features to represent and characterize each digit (such as HOG, Zernike Moments, etc), we’ll instead use just the raw, grayscale pixel intensities of the image.
Step 4 – Training our classification model: Our k-NN classifier will be trained on the raw pixel intensities of the images in the training set. We’ll then determine the best value of k using the validation set.
Step 5 – Evaluating our classifier: Once we have found the best value of k, we can then evaluate our k-NN classifier on our testing set.

# How To Run ? 
Since, this is a jupyter notebook file so, make sure that you have Anaconda software installed on your respective systems to run this source code file.

After installation you just need to upload this file after downloading it from the repository in Jupyter Notebook's local directory and run the notebook cells accordingly after uploading the training and testing datasets in the local directory of Jupyter Notebook as well.

# Summary
In this project, I learnt about the most simple machine learning classifier — the k-Nearest Neighbor classifier, or simply k-NN for short. The k-NN algorithm classifies unknown data points by comparing the unknown data point to each data point in the training set. This comparison is done using a distance function or similarity metric. Then, from the k most similar examples in the training set, we accumulate the number of “votes” for each label. The category with the highest number of votes “wins” and is chosen as the overall classification.

While simple and intuitive, and though it can even obtain very good accuracy in certain situations, the k-NN algorithm has a number of drawbacks. The first is that it doesn’t actually “learn” anything — if the algorithm makes a mistake, it has no way to “correct” and “improve” itself for later classifications. Secondly, without specialized data structures, the k-NN algorithm scales linearly with the number of data points, making it a questionable choice for large datasets.

# To conclude, we applied the k-NN algorithm to the MNIST dataset for handwriting recognition. Simply by computing the Euclidean distance between raw pixel intensities, we were able to obtain a very high accuracy of 98%. However, it’s important to note that the MNIST dataset is heavily pre-processed, and we will require more advanced methods for recognize handwriting in real-world images.
