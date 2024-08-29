import numpy as np
import pandas as pd

data = pd.read_csv("../csv/SpotifyFeatures.csv",comment = '#', delimiter=',', header=0)
#print(data.shape) #232725 songs and 18 features 
genre = np.array(data['genre'].shape)
print(genre) #232725 songs
