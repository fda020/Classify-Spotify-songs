import numpy as np
import pandas as pd
import display

display.dataframe_display()

all_df = pd.read_csv("../csv/SpotifyFeatures.csv",comment = '#', delimiter=',', header=0)
print(all_df.shape) #232725 songs and 18 features 


genre = np.array(all_df['genre'])

#Filtering the dataset to only give Classical
Classical_df = all_df.loc[all_df['genre'] == 'Classical']
Classical_df.index = [0] * len(Classical_df)
print(Classical_df.shape) #9256 Classical songs

#Filtering the dataset to only give pop 
Pop_df = all_df.loc[all_df['genre'] == 'Pop']
Pop_df.index = [1] * len(Pop_df)
print(Pop_df.shape) #9386 Pop songs

#Put the dataframe for pop and classical together
CP_df = pd.concat([Classical_df, Pop_df])
print(CP_df)

#With only liveness and loudness
CP_shorten = CP_df[['liveness', 'loudness']]
print(CP_shorten)
