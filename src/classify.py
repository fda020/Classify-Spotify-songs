import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import display

class DataFrame():
    def __init__(self, datafile):
        self.all_df = pd.read_csv(datafile,comment = '#', delimiter=',', header=0)

    def filtering(self, heading, filter, ind):
        self.filter_df = self.all_df.loc[self.all_df[heading] == filter]
        self.filter_df.index = [ind] * len(self.filter_df)

    def retrieve_matrix(self):
        return np.transpose(np.array([self.filter_df['liveness'],self.filter_df['loudness']]))
    
    def retrieve_vector(self):
        return np.array(self.filter_df.index)
    
class train_and_test():
    def __init__(self, matrix, vector):
        combine = np.c_[vector, matrix]
        np.random.shuffle(combine)
        print(combine)

        #make the matrix and the vector by splitting
        split_matrix = combine[:, 1:]
        split_vector = combine[:, 0]
        print(split_matrix.shape)
        print(split_vector.shape)

        split_ratio = 0.8
        split_index = int(split_ratio * len(split_matrix))

        #Make a train and test set for the matrix and the vector
        self.matrix_train = split_matrix[:split_index]
        self.vector_train = split_vector[:split_index]

        self.matrix_test = split_matrix[split_index:]
        self.vector_test = split_vector[split_index:]

# bruker den til å train data også bruke test data for å teste modellen som er laget
class Log_Reg():
    def __init__(self, matrix):
        self.matrix = matrix
        self.num_sample = len(self.matrix)
        self.num_features = len(self.matrix[0])

        print(self.num_features, self.num_sample)
        self.weight = np.zeros(self.num_features)
        self.bias = 0
        

    def sigmoid(self, x): #here x train 
        z = np.matmul(x, self.weight) + self.bias
        y_pred = 1/(1 + np.exp(-z))
        return y_pred
    
    def loss(self, y_true, y_pred): # y pred er y_pred (fra sigmoid) og y true er y train vector
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def stoch_grad_desc(self, x, y, learning_rate, num_epochs):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.training_error = []

        for i in range(self.num_epochs):
            for j in range(self.num_sample):
                x_new = x[i]
                y_new = y[j]

                #the sigmoid function
                # kan brule y_pred = self.sigmoid(x_new)
                z = np.matmul(x_new, self.weight) + self.bias
                y_pred = 1/(1 + np.exp(-z))

                #the gradients for weight and bias
                self.gradients_weight = (y_pred - y_new) * x_new
                self.gradients_bias = y_pred - y_new

                self.weight -= self.learning_rate * self.gradients_weight
                self.bias -= self.learning_rate * self.gradients_bias

        



if __name__== "__main__":
    display.dataframe_display()
    dataframe = DataFrame("../csv/SpotifyFeatures.csv")
    print(dataframe.all_df.shape) #232725 songs and 18 features 

    Classical_df = DataFrame("../csv/SpotifyFeatures.csv")
    Classical_df.filtering('genre', 'Classical', 0) 
    print(Classical_df.filter_df.shape) #9256 Classical songs

    Pop_df = DataFrame("../csv/SpotifyFeatures.csv")
    Pop_df.filtering('genre', 'Pop', 1) 
    print(Pop_df.filter_df.shape) #9386 Pop songs
    
    # Put the dataframes together
    CP_df = DataFrame("../csv/SpotifyFeatures.csv")
    CP_df.filter_df = pd.concat([Classical_df.filter_df, Pop_df.filter_df])
    #Make dataframe to only view liveness and loudness 
    CP_df.filter_df = CP_df.filter_df[['liveness', 'loudness']]
    print(CP_df.filter_df)

    song_matrix = CP_df.retrieve_matrix()
    print(song_matrix)

    genre_vector = CP_df.retrieve_vector() 
    print(genre_vector)
    
    train = train_and_test(song_matrix, genre_vector)
    matrix_train = train.matrix_train
    vector_train = train.vector_train

    matrix_test = train.matrix_test
    vector_test = train.vector_test

    print(matrix_train)
    print(vector_train)

    model = Log_Reg(matrix_train)

    model.stoch_grad_desc(matrix_train, vector_train, 0.02, 40)

    

    
    
