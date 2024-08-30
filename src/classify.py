import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import display

display.dataframe_display()

all_df = pd.read_csv("../csv/SpotifyFeatures.csv",comment = '#', delimiter=',', header=0)
print(all_df.shape) #232725 songs and 18 features 


#Filtering the dataset to only give Classical in a dataframe
Classical_df = all_df.loc[all_df['genre'] == 'Classical']
Classical_df.index = [0] * len(Classical_df)
print(Classical_df.shape) #9256 Classical songs

#Filtering the dataset to only give Pop in a dataframe
Pop_df = all_df.loc[all_df['genre'] == 'Pop']
Pop_df.index = [1] * len(Pop_df)
print(Pop_df.shape) #9386 Pop songs

#Put the dataframe for pop and classical together
CP_df = pd.concat([Classical_df, Pop_df])
print(CP_df)

#To make the dataframe smaller with genre, artist name, track name, track id, liveness and loudness
CP_shorten = CP_df[['genre', 'artist_name', 'track_name', 'track_id','liveness', 'loudness']]
print(CP_shorten)

#Matrix where rows are songs and the two columns are liveness and loudness
songs = np.transpose(np.array([CP_shorten['liveness'],CP_shorten['loudness']]))
print(songs)
#The vector as labels
genre = np.array(CP_shorten.index)
print(genre)

#Combine to shuffle all data before making a test and training set
combine = np.c_[genre, songs]
np.random.shuffle(combine)
print(combine)

#make the matrix and the vector by overwriting 
songs = combine[:, 1:]
genre = combine[:, 0]
print(songs.shape)
print(genre.shape)

#Make a train and test set for the matrix and the vector
split_ratio = 0.8
split_index = int(split_ratio * len(songs))

songs_train = songs[:split_index]
genre_train = genre[:split_index]

songs_train = songs[split_index:]
genre_train = genre[split_index:]
