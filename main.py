import os
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pickle
import numpy as np
import july
import matplotlib.pyplot as plt




client_id = os.environ.get('SPOTIFY_CLIENT_ID')
client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET')
client_credentials_manager = SpotifyClientCredentials(
    client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


#First we get the listening history
df = pd.read_json('StreamingHistory0.json')
df_temp = pd.read_json('StreamingHistory1.json')
df = pd.concat([df,df_temp])
df['Track'] = df['artistName']+'-'+df['trackName']

#We keep the date whitout the hour
df["endTime"]=df["endTime"].apply(lambda x:x[0:10])
del df_temp

musics=pd.DataFrame(df['Track'].unique())
df_audio_features = pd.DataFrame()
fail = []

for music in musics[0]:
    results = sp.search(q=music, type="track")
    if len(results)!=0:
        track_uri = results["tracks"]["items"][0]["uri"]

        #We use Spotify's feature
        features = sp.audio_features(tracks=[track_uri])[0]
        print(features)
        if type(features) !='NoneType':

            df_audio_features = df_audio_features.append(features, ignore_index=True)
        else :
            fail.append(music)
    else :
        fail.append(music)

features=["danceability","acousticness","energy","instrumentalness","liveness","valence","loudness","speechiness"]
df.drop("mode",axis=1)
df[df.columns[6:16]]=df[features]


with open('model.pickle', 'rb') as f:
    pipe = pickle.load(f)
    
with open('labelencoder.pickle', 'rb') as f:
    le = pickle.load(f)

#We predict the mood of each song
mood = pipe.predict(df[features])
df['mood']=le.inverse_transform(mood)
df['mood_int']=mood

#We group by day keeping the mode
mood_list = df.groupby('endTime')['mood_int'].apply(lambda x: x.mode()[0])
df

#We plot the result
july.heatmap(mood_list.index, mood_list.values, title='User\'s Mood', cmap="Dark2",month_grid=True,value_label=False)
'''
0 : calm
1 : energetic
2 : happy
3 : sad
'''
