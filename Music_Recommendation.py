import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')

tracks = pd.read_csv('top10s.csv')
# tracks.head()

# tracks.shape

# tracks.info()

tracks.isnull().sum().plot.bar()
# plt.show()

tracks = tracks.drop(['Unnamed: 0', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17'], axis = 1)

# tracks.info()

a = tracks.drop(['title', 'artist', 'top genre'], axis = 1)

model = TSNE(n_components = 2, random_state = 0)
tsne_data = model.fit_transform(a.head(500))
plt.figure(figsize = (7, 7))
plt.scatter(tsne_data[:,0], tsne_data[:,1])
# plt.show()
# sb.relplot(data=tracks, x="top genre", y="pop", hue="year")

tracks['title'].nunique(), # tracks.shape

tracks = tracks.sort_values(by=['pop'], ascending=False)
tracks.drop_duplicates(subset=['title'], keep='first', inplace=True)

plt.figure(figsize = (10, 5))
sb.countplot(tracks['year'])
plt.axis('off')
# plt.show()

float = []
for col in a.columns:
    if tracks[col].dtype == 'float64':
        float.append(col)

# len(float)

plt.subplots(figsize = (15, 5))
for i, col in enumerate(float):
    plt.subplot(2, 5, i + 1)
    sb.distplot(tracks[col])
plt.tight_layout()
# plt.show()

int = []
for col in a.columns:
    if a[col].dtype == 'int64':
        int.append(col)

# len(int)

plt.subplots(figsize = (15, 5))
for i, col in enumerate(int):
    plt.subplot(2, 5, i + 1)
    sb.distplot(tracks[col])
plt.tight_layout()
# plt.show()

# By default, %%capture discards these streams. This is a simple way to suppress unwanted output.
song_vectorizer = CountVectorizer()
t = song_vectorizer.fit(tracks['top genre'])

tracks = tracks.sort_values(by=['pop'], ascending=False).head(603)

def get_similarities(song_name, data):

# Getting vector for the input song.
    text_array1 = t.transform(data[data['title']==song_name]['top genre']).toarray()
    num_array1 = data[data['title']==song_name].select_dtypes(include=np.number).to_numpy()

# We will store similarity for each row of the dataset.
    sim = []
    for idx, row in data.iterrows():
        name = row['title']

        # Getting vector for current song.
        text_array2 = t.transform(data[data['title']==name]['top genre']).toarray()
        num_array2 = data[data['title']==name].select_dtypes(include=np.number).to_numpy()

        # Calculating similarities for text as well as numeric features
        text_sim = cosine_similarity(text_array1, text_array2)[0][0]
        num_sim = cosine_similarity(num_array1, num_array2)[0][0]
        sim.append(text_sim + num_sim)

    return sim


def recommend_songs(song_name, data=tracks):
    if tracks[tracks['title'] == song_name].shape[0] == 0:
        print('This song is either not so popular or you have entered invalid_name.\n Some songs you may like:\n')

        for song in data.sample(n=5)['title'].values:
            print(song)
        return

    data['similarity_factor'] = get_similarities(song_name, data)

    data.sort_values(by=['similarity_factor', 'pop'],
                    ascending = [False, False],
                    inplace=True)

    print(data[['title', 'artist']][2:7])


# a = input("What is your favourite music: ")
# recommend_songs(a)