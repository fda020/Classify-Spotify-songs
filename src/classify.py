import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import display

class DataFrame():
    def __init__(self, datafile):
        self.all_df = pd.read_csv(datafile, delimiter=',', header=0)

    def filtering(self, heading, filter, ind):
        self.all_df = self.all_df.loc[self.all_df[heading] == filter]
        self.all_df['label'] = ind
        
    def retrieve_matrix(self):
        return np.transpose(np.array([self.all_df['liveness'],self.all_df['loudness']]))
    
    def retrieve_vector(self):
        return np.array(self.all_df['label'])
    
class train_and_test():
    def __init__(self, matrix, vector):
        #need to combine before shuffling the data
        combine = np.c_[vector, matrix]
        np.random.shuffle(combine)
        #print(combine)

        #make the matrix and the vector by splitting
        split_matrix = combine[:, 1:]
        split_vector = combine[:, 0]
        # print(split_matrix)
        # print(split_vector)

        #set the split ratio. Here we have 80 20%
        split_ratio = 0.8
        split_index = int(split_ratio * len(split_matrix))

        #Make a train and test set for the matrix and the vector
        self.matrix_train = split_matrix[:split_index]
        self.vector_train = split_vector[:split_index]

        self.matrix_test = split_matrix[split_index:]
        self.vector_test = split_vector[split_index:]


class Log_Reg():
    def __init__(self, matrix):
        self.matrix = matrix
        #number of samples and features
        self.num_sample = len(self.matrix)
        self.num_features = len(self.matrix[0])

        #add a weight so the model can learn during training 
        self.weight = np.zeros(self.num_features)
        self.bias = 0
        
    
    def loss(self, y_true, y_pred): 
        epsilon = 0.0000001 #to avoid log(0)
        #use clipping to prevent log(0) or log(1), so a number inbetween
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon) 
        #binary cross entropy loss formula
        loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)) 
        return loss
    
    #preforms stochasitc gradient descent to train the regression model
    def stoch_grad_desc(self, x, y, learning_rate, num_epochs):
        #the learning rate to test with 0.01 or 0.001
        self.learning_rate = learning_rate 
        self.num_epochs = num_epochs 

        #where the training error is stored before plotting
        self.training_error = [] 
        
        for i in range(self.num_epochs):
            for j in range(self.num_sample):
                x_new = x[j]
                y_new = y[j]

                #the sigmoid function to find the predicted y
                z = np.matmul(x_new, self.weight) + self.bias
                y_pred = 1/(1 + np.exp(-z))

                #the gradients for weight and bias
                self.gradients_weight = (y_pred - y_new) * x_new
                self.gradients_bias = y_pred - y_new

                #this is updated to minimize loss
                self.weight -= self.learning_rate * self.gradients_weight
                self.bias -= self.learning_rate * self.gradients_bias
            
            #this is to evaluate the model
            z = np.matmul(x, self.weight) + self.bias
            y_pred_acc = 1 / (1 + np.exp(-z))
            y_pred_bin = y_pred_acc >= 0.5

            loss = self.loss(y, y_pred_acc) #loss is computed after each epoch 
            self.training_error.append(loss)

        accuracy = np.mean(y_pred_bin == y) 

        print('training accuracy', accuracy)
        #print(self.bias)

    def prediction(self, test_X):
        z = np.matmul(test_X, self.weight) + self.bias 
        y_pred = 1/(1 + np.exp(-z)) >= 0.5
        return y_pred
            
def confusion_matrix(vector_test, y_pred_test):
    tp = np.sum(((vector_test == 1)) & (y_pred_test == 1)) #true positive
    tn = np.sum(((vector_test == 0)) & (y_pred_test == 0)) #true negative
    fp = np.sum(((vector_test == 0)) & (y_pred_test == 1)) #false positive
    fn = np.sum(((vector_test == 1)) & (y_pred_test == 0)) #false negative

    conf_matrix = np.array([[tp, fp], [fn, tn]])
    print(conf_matrix)



if __name__== "__main__":
    display.dataframe_display()
    dataframe = DataFrame("../csv/SpotifyFeatures.csv")
    print(dataframe.all_df.shape) #232725 songs and 18 features 
    # print(dataframe.all_df)

    Classical_df = DataFrame("../csv/SpotifyFeatures.csv")
    Classical_df.filtering('genre', 'Classical', 0) 
    print(Classical_df.all_df.shape) #9256 Classical songs

    Pop_df = DataFrame("../csv/SpotifyFeatures.csv")
    Pop_df.filtering('genre', 'Pop', 1)
    print(Pop_df.all_df.shape) #9386 Pop songs
    
    # Put the dataframes together
    CP_df = DataFrame("../csv/SpotifyFeatures.csv")
    CP_df.all_df = pd.concat([Classical_df.all_df, Pop_df.all_df])
    #Make dataframe to only view liveness and loudness 
    CP_df.all_df = CP_df.all_df[['liveness', 'loudness', 'label']]
    # print(CP_df.all_df)

    song_matrix = CP_df.retrieve_matrix()
    # print(song_matrix)

    genre_vector = CP_df.retrieve_vector() 
    # print(genre_vector)
    
    train = train_and_test(song_matrix, genre_vector)
    matrix_train = train.matrix_train
    vector_train = train.vector_train

    matrix_test = train.matrix_test
    vector_test = train.vector_test

    #print(matrix_train)
    #print(vector_train)

    #the train matrix is used in the model to train it
    model = Log_Reg(matrix_train)
    model.stoch_grad_desc(matrix_train, vector_train, 0.001, 100)

    plt.plot(model.training_error, color = 'blue')
    plt.xlabel('Epoch')
    plt.ylabel('Training Error')
    plt.show()

    test_model = model.prediction(matrix_test)
    test_acc = np.mean(test_model == vector_test)
    print('test accuracy', test_acc)

    conf_matrix = confusion_matrix(vector_test, test_model)
