import pandas as pd
import streamlit as st
import pickle


tampa = pickle.load(open('tampa.pd', 'rb'))
apartments = tampa.name_x.unique()
apartments = sorted([i for i in apartments])

st.write("""
## Faster  Apt - Find Your Dream Apartment in Tampa, Florida!""")

from PIL import Image
image = Image.open('tampa_img.jpeg')
st.image(image, width=750)



st.sidebar.header('SEARCH FOR APARTMENTS:')
name = st.sidebar.selectbox('SELECT AN APARTMENT', apartments)

data = tampa[tampa.name_x == name]
fake_data = data[data.fake_all==1]

st.title(name)
st.write(""" ### Review Stars: """)
st.subheader(round(sum(data['stars_x'])/len(data['stars_x']), 1))

st.write(""" ### Suspicious Reviews:""")
st.subheader(str(round(len(fake_data) / len(data) * 100, 1)) + '%')

data2 = data[data.fake_all==0]
data3 = data2[data2['neutr']==0]
stars = data3['good']
text = data3['text']

#number = [1,2,3,4,5]
#num = st.sidebar.selectbox('SELECT NUMBER OF REVIEW WORDS', number, 2)
num = int(st.sidebar.number_input('SELECT NUMBER OF REVIEW WORDS', int(1), int(5), int(3)))

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class ToText(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X2 = X[X['neutr'] == 0]
        text = [i for i in X2['text']]
        return text

from sklearn.feature_extraction.text import TfidfVectorizer

ng_tfidf = TfidfVectorizer(ngram_range=(num,num), stop_words='english')
tdi_text = ng_tfidf.fit(text)

from sklearn.pipeline import Pipeline
vect_text = Pipeline([
    ('to_text', ToText()),
    ('tfidfvectorizer', TfidfVectorizer(ngram_range=(num,num), stop_words='english')),
])

vectorized = vect_text.fit_transform(data2)

from sklearn.naive_bayes import MultinomialNB
bayes_model = MultinomialNB().fit(vectorized, stars)
log_prob = bayes_model.feature_log_prob_
polarity1 = log_prob[1]/log_prob[0]
words = ng_tfidf.get_feature_names()
words_probs = list(zip(words, polarity1))
words_probs = sorted(words_probs, key=lambda x: x[1], reverse=True)
polarity2 = log_prob[0]/log_prob[1]
words_probs2 = list(zip(words, polarity2))
words_probs2 = sorted(words_probs2, key=lambda x: x[1], reverse=True)
pos_words, neg_words = words_probs2, words_probs

pos_words = [pos_words[i][0].upper() for i in range(len(pos_words))]
neg_words = [neg_words[i][0].upper() for i in range(len(neg_words))]

pos_n = st.sidebar.slider('NUMBER OF POSITIVE REVIEWS', 1, 30, 10)
neg_n = st.sidebar.slider('NUMBER OF NEGATIVE REVIEWS', 1, 30, 10)
ntr_n = st.sidebar.slider('NUMBER OF NEUTRAL REVIEWS', 1, 30, 10)

col1, col2, col3 = st.columns(3)

len_pos =  round(len(data2[data2['good'] == 1]) /(len(data2[data2['good'] == 1]) + len(data2[data2['good']==0])) * 100, 1)

col1.write(""" ### Positive Reviews """)
col1.subheader(str(len_pos) + '%')
col1.write(pos_words[:pos_n])

len_neg =  round(len(data2[data2['bad'] == 1]) /(len(data2[data2['bad'] == 1]) + len(data2[data2['bad']==0])) * 100, 1)

col2.write(""" ### Negative Reviews""")
col2.subheader(str(len_neg)+'%')
col2.write(neg_words[:neg_n])


data4 = data2[data2.neutr==1]
text2 = data4.text
from sklearn.feature_extraction.text import CountVectorizer
neutr_words = CountVectorizer(max_features=30,
                             ngram_range=(num,num),
                             stop_words='english')
def vector():
    if len(text2) > 0:
        ntr_words = neutr_words.fit(text2)
        ntr_words = ntr_words.get_feature_names()
        ntr_words = [ntr_words[i].upper() for i in range(len(ntr_words))]
        return ntr_words[:ntr_n]
    else:
        return ''


len_ntr = round(len(data2[data2['neutr'] == 1]) /(len(data2[data2['neutr'] == 1]) + len(data2[data2['neutr']==0])) * 100, 1)


col3.write(""" ### Neutral Reviews  """)

col3.subheader(str(len_ntr)+'%')

col3.write(vector())

pos_text_data = data[data.good==1]
post_text = pos_text_data['text']
post_text = [i for i in post_text]

neg_text_data = data[data.bad==1]
neg_text = neg_text_data['text']
neg_text = [i for i in neg_text]

ntr_text_data = data[data.neutr==1]
ntr_txt = ntr_text_data['text']
ntr_txt = [i for i in ntr_txt]

A = ['NO', 'YES']
pos_reviews = st.sidebar.selectbox('READ ALL POSITIVE REVIEWS', A)
neg_reviews = st.sidebar.selectbox('READ ALL NEGATIVE REVIEWS', A)
neutr_reviews = st.sidebar.selectbox('READ ALL NEUTRAL REVIEWS', A)

class ReadReviews():

    def positive_reviews(self):
        return st.write(post_text) if pos_reviews == 'YES' else ''

    def negative_reviews(self):
        return st.write(neg_text) if neg_reviews == 'YES' else ''

    def neutral_reviews(self):
        return st.write(ntr_txt) if neutr_reviews == 'YES' else ''

ReadReviews().positive_reviews()
ReadReviews().negative_reviews()
ReadReviews().neutral_reviews()


st.write("""
Developed by Samad Karimov

For questions and/or comments: samkerim88@gmail.com""")