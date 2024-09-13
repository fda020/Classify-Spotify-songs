# Classify Spotify Songs
The code provided gives a simple classification of Spotify songs from a CSV file using Machine Learning.
It uses a logistic regression model with stochastic gradient descent.
First, the dataset is split into a vector with labels, where 'Pop' = 1 and 'Classical' = 0. 
Then, it creates a matrix with only the 'loudness' and 'liveness' features, instead of using all available features.
The data is divided into a train and test set, with an 80%-20% split, before creating and training the model. 
The loss is calculated using the binary cross-entropy formula and plotted for visualization.
Finally, the accuracy for both the training and test sets is computed, and a confusion matrix is generated to show the model's performance on the test set.

How to run code:
- Go to src in terminal
- Type in python3 classify.py 
