from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import concatenate, Input, multiply, subtract, average
import keras.backend as K
from keras.layers import Lambda
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation, GRU, Bidirectional
from keras import regularizers
from keras import losses
import keras
from keras import optimizers
from keras import *
from layer2 import AttentionWithContext, Addition
import io
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn import svm
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('./all.csv') #load dataset

#function to remove ounctuation
def removePunctuation(x):
    x = x.lower()
    x = re.sub(r'[^\x00-\x7f]',r' ',x)
    x = x.replace('\r','')
    x = x.replace('\n','')
    x = x.replace('  ','')
    x = x.replace('\'','')
    return re.sub("["+string.punctuation+"]", " ", x)


#getting stop words
from nltk.corpus import stopwords
nltk.download('stopwords')
stops = set(stopwords.words("english")) 


#function to remove stopwords
def removeStopwords(x):
    filtered_words = [word for word in x.split() if word not in stops]
    return " ".join(filtered_words)


def processText(x):
    x= removePunctuation(x)
    x= removeStopwords(x)
    return x

import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
X= pd.Series([word_tokenize(processText(x)) for x in df['content']])
X.head()

#vectorizing X and y to process
vectorize=CountVectorizer(max_df=0.95, min_df=0.005)
X=vectorize.fit_transform(df['content'], df['author'])
transform = {
    "Mythology & Folklore" : 2,
    "Nature" : 1,
    "Love" :  0,
}
y = []
for typ in df.type:
    y.append(transform[typ])
y = np.array(y)

fasttext_vectors_file = "C:/Users\saman\web_data\embeddings\crawl-300d-2M-subword\crawl-300d-2M-subword.vec"


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    i = 0
    for line in fin:
        i+=1
        if i % 100000 == 0 :
            print (i)
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data

fasttext_map = load_vectors(fasttext_vectors_file)

def sentence2sequence(sentence):
    """
     
    - Turns an input sentence into an (n,d) matrix, 
        where n is the number of tokens in the sentence
        and d is the number of dimensions each word vector has.
    
      Tensorflow doesn't need to be used here, as simply
      turning the sentence into a sequence based off our 
      mapping does not need the computational power that
      Tensorflow provides. Normal Python suffices for this task.
    """
    tokens = sentence.lower().split() # changed from split(" ") to split()
    rows = []
    words = []
    #Greedy search for tokens
    for token in tokens:
        i = len(token)
        while len(token) > 0 :
            word = token[:i]
            if word in fasttext_map:
                rows.append(list(fasttext_map[word]))
                words.append(word)
                token = token[i:]
                i = len(token)
            else:
                i = i-1
            if i < 0 :
                print (token, " coudn't find the word")
                break
    return rows, words

def fit_to_size(matrix, shape):
    res = np.zeros(shape)
    slices = [slice(0,min(dim,shape[e])) for e, dim in enumerate(matrix.shape)]
    res[slices] = matrix[slices]
    return res

max_length = 30
vector_size = 300

def split_data_into_scores(Data):
    sentences = []
    y = []
    i = 0
    for index, row in Data.iterrows():
        if i % 50 == 0:
            print (i, '/ ', len(Data))
        
        snts = sentence2sequence(row['content'].lower())[0]
        while len(snts) != 0 :
            
            t = [0,0,0]
            t[transform[row['type']]] = 1
            y.append(t)

            sentences.append(np.vstack(snts[:max_length]))
            
            if len(snts) > max_length:
                snts = snts[max_length:]
            else : break
                
        i+=1
        
    
    sentences = np.stack([fit_to_size(x, (max_length, vector_size))
                        for x in sentences])
                                
    return sentences, np.array(y)
    
def data_to_score_one_row(row):
    sentences = []
    y = []
    snts = sentence2sequence(row['content'].lower())[0]
    while len(snts) != 0 :
        t = [0,0,0]
        t[transform[row['type']]] = 1
        y.append(t)

        sentences.append(np.vstack(snts[:max_length]))

        if len(snts) > max_length:
            snts = snts[max_length:]
        else : break
    sentences = np.stack([fit_to_size(x, (max_length, vector_size))
                        for x in sentences])
    return sentences, np.array(y)


msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

data_feature_list, y = split_data_into_scores(train)
X_train, X_valid, y_train, y_valid = train_test_split(
    data_feature_list, y, test_size=0.125, random_state=42)

adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)

model = Sequential()

num_layers = 1
is_attention = True
is_bidirectional = True
hidden_units = 64
vector_length = 300
num_classes = 3
for i in range(num_layers):
    return_sequences = is_attention or (num_layers > 1 and i < num_layers-1)

    if is_bidirectional:
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=return_sequences, dropout=0.2, kernel_initializer=initializers.glorot_normal(seed=777), bias_initializer='zeros', input_shape=(max_length, vector_length))))
    else:
        model.add(LSTM(hidden_units, return_sequences=return_sequences, dropout=0.2, kernel_initializer=initializers.glorot_normal(seed=777), bias_initializer='zeros'))

    if is_attention:
        model.add(AttentionWithContext())
        model.add(Addition())

model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_normal(seed=777), bias_initializer='zeros'))
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=["accuracy"])
model.summary()



model.fit(X_train, np.array(y_train), batch_size = 128, epochs = 200, validation_data=(X_valid, np.array(y_valid)))

def test_function(model, test_d):
    pred = []
    y_true = []
    for index, row in test_d.iterrows():
        if len(pred) % 100 == 0:
            print (len(pred))
        pred_t = []
        snt, y = data_to_score_one_row(row)
        y_true.append(transform[row['type']])
        for chunk in snt:
            tt= model.predict(np.array([chunk]))
            pred_t.append(np.argmax(tt[0]))
        pred.append(np.argmax(np.bincount(pred_t)))
    y_true = np.array(y_true)
    from sklearn.metrics import accuracy_score
    return accuracy_score(pred, y_true)

test_function(model, test)



