import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

df = pd.read_csv('taylor_swift_lyrics.csv', encoding='latin-1')
df.head()

songs = df.groupby('track_title').agg({'lyric': lambda x: ' '.join(x), 'year': 'mean'}).reset_index()
songs.head()


pd.options.display.max_colwidth = 5000

songs.sort_values('year')


nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(
    ['love', 'baby', 'go', 'time', 'bad', 'got', 'beautiful', 'let', 'never', 'life', 'oh', 'girl', 'one', 'knew',
     'tell', 'never', 'dress', 'car', 'yeah', 'placeknow', 'better', 'want', 'like', 'run', 'see', 'best', 'everything',
     'trying', 'going', 'ever', 'stay', 'never', 'gonna', 'hold', 'getting', 'remember', 'time', 'still', 'last',
     'wanna', 'dancing', 'like', 'would', 'come', 'good', 'first', 'end', 'new', 'feeling', 'think', 'back', 'said',
     'home', 'gone', 'get', 'things', 'come', 'say', 'song'])

vectorizer = TfidfVectorizer(stop_words=stop_words, min_df=0.1)
tfidf = vectorizer.fit_transform(songs['lyric'])

nmf = NMF(n_components=6)
topic_values = nmf.fit_transform(tfidf)

for topic_num, topic in enumerate(nmf.components_):
    message = 'Topic #{}:'.format(topic_num + 1)
    message += " ".join([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-11:-1]])
    print(message)

st = "love baby go time bad got beautiful let never life oh girl one knew tell never dress car yeah place know better " \
     "want like run see best everything trying going ever stay never gonna hold getting remember time still last " \
     "wanna dancing like would come good first end new feeling think back said home gone get things come say song "

topic_labels = ['love', 'memories', 'breakups', 'party', 'homesick', 'independence']
df_topics = pd.DataFrame(topic_values, columns=topic_labels)

songs = songs.join(df_topics)

songs.loc[songs['love'] >= 0.1, 'love'] = 1
songs.loc[songs['memories'] >= 0.1, 'memories'] = 1
songs.loc[songs['breakups'] >= 0.1, 'breakups'] = 1
songs.loc[songs['party'] >= 0.1, 'party'] = 1
songs.loc[songs['independence'] >= 0.1, 'independence'] = 1

songs.loc[songs['love'] <= 0.1, 'love'] = 0
songs.loc[songs['memories'] <= 0.1, 'memories'] = 0
songs.loc[songs['breakups'] <= 0.1, 'breakups'] = 0
songs.loc[songs['party'] <= 0.1, 'party'] = 0
songs.loc[songs['independence'] <= 0.1, 'independence'] = 0

year_topics = songs.groupby('year').sum().reset_index()

plt.figure(figsize=(20, 10))
plt.plot(year_topics['year'], year_topics['love'], label='love')
plt.plot(year_topics['year'], year_topics['memories'], label='memories')
plt.plot(year_topics['year'], year_topics['breakups'], label='breakups')
plt.plot(year_topics['year'], year_topics['party'], label='party')
plt.plot(year_topics['year'], year_topics['homesick'], label='homesick')
plt.plot(year_topics['year'], year_topics['independence'], label='independence')

plt.legend()
plt.show()
