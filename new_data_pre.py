import re
import codecs

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical

import numpy as np

MAX_SENT_LENGTH = 80
MAX_SENTS = 150
MAX_NB_WORDS = 30000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
CAR_TRAIN_PATH = './data/car_train.txt'
CAR_TEST_PATH = './data/car_test.txt'
PET_TRAIN_PATH = './data/pet_train.txt'
PET_TEST_PATH = './data/pet_test.txt'
categ = 'car'
car_train_reviews = []
car_train_labels = []

file = codecs.open(CAR_TRAIN_PATH, "r", encoding='utf-8')
for line in file.readlines():
    terms = line.split('\t')
    query = terms[1].lower().replace(',', ' , ').replace('...', ' ... ').replace('/', ' / ').replace('.',
                                                                                                     ' . ').replace('&',
                                                                                                                    ' & ').replace(
        '@', ' @ ').replace('!', ' ! ').replace('?', ' ? ').replace('[', ' [ ').replace(']', ' ] ').replace('(',
                                                                                                            ' ( ').replace(
        ')', ' ) ')

    title = terms[2].lower().replace(',', ' , ').replace('...', ' ... ').replace('/', ' / ').replace('.',
                                                                                                     ' . ').replace('&',
                                                                                                                    ' & ').replace(
        '@', ' @ ').replace('!', ' ! ').replace('?', ' ? ').replace('[', ' [ ').replace(']', ' ] ').replace('(',
                                                                                                            ' ( ').replace(
        ')', ' ) ')
    text = title + "#" + query
    sentences = text.split('#')

    car_train_reviews.append([x.split() for x in sentences])
    # print(car_train_reviews)
    car_train_labels.append(int(terms[0]))
file.close()

car_test_reviews = []
car_test_labels = []

file = codecs.open(CAR_TEST_PATH, "r", encoding='utf-8')
for line in file.readlines():
    terms = line.split('\t')
    query = terms[1].lower().replace(',', ' , ').replace('...', ' ... ').replace('/', ' / ').replace('.',
                                                                                                     ' . ').replace('&',
                                                                                                                    ' & ').replace(
        '@', ' @ ').replace('!', ' ! ').replace('?', ' ? ').replace('[', ' [ ').replace(']', ' ] ').replace('(',
                                                                                                            ' ( ').replace(
        ')', ' ) ')

    title = terms[2].lower().replace(',', ' , ').replace('...', ' ... ').replace('/', ' / ').replace('.',
                                                                                                     ' . ').replace('&',
                                                                                                                    ' & ').replace(
        '@', ' @ ').replace('!', ' ! ').replace('?', ' ? ').replace('[', ' [ ').replace(']', ' ] ').replace('(',
                                                                                                            ' ( ').replace(
        ')', ' ) ')
    text = title + "#" + query
    sentences = text.split('#')

    car_test_reviews.append([x.split() for x in sentences])
    # print(car_train_reviews)
    car_test_labels.append(int(terms[0]))
file.close()

# In[2]:


word_dict = {'PADDING': [0, 999999], 'UNK': [1, 99999]}

for i in car_train_reviews:
    for sent in i:
        for word in sent:
            if not str(word) in word_dict.keys():
                word_dict[str(word)] = [len(word_dict), 1]
            else:
                word_dict[str(word)][1] += 1

for i in car_test_reviews:
    for sent in i:
        for word in sent:
            if not str(word) in word_dict.keys():
                word_dict[str(word)] = [len(word_dict), 1]
            else:
                word_dict[str(word)][1] += 1

import pickle
f = open('./data/car_dict.pkl', 'wb')
pickle.dump(word_dict, f)

embdict = dict()

# all_emb=[]
plo = 0
print('reading')
with open('./code/GoogleNews-vectors-negative300.txt', 'r')as f:
    j = f.readlines()
    for line in range(len(j)):
        k = j[line].split()
        word = k[0]
        if len(word) != 0:
            tp = [float(x) for x in k[1:]]
            if word in word_dict:
                embdict[word] = tp
                print(plo, line, word)
                plo += 1

from numpy.linalg import cholesky

print(len(embdict), len(word_dict))
print(len(word_dict))
lister = [0] * len(word_dict)
xp = np.zeros(300, dtype='float32')

cand = []

for i in embdict.keys():
    lister[word_dict[i][0]] = np.array(embdict[i], dtype='float32')
    cand.append(lister[word_dict[i][0]])
cand = np.array(cand, dtype='float32')

mu = np.mean(cand, axis=0)
Sigma = np.cov(cand.T)
# R = cholesky(Sigma)

norm = np.random.multivariate_normal(mu, Sigma, 1)
print(mu.shape, Sigma.shape, norm.shape)

for i in range(len(lister)):
    if type(lister[i]) == int:
        lister[i] = np.reshape(norm, 300)
lister[0] = np.zeros(300, dtype='float32')
lister = np.array(lister, dtype='float32')
print(lister.shape)

f = open('./data/car_embedding.pkl', 'wb')
pickle.dump(embdict, f)

maxlen = 80
car_train_data = []
for i in car_train_reviews:
    review = []
    for sent in i:
        sentence = []
        for word in sent:
            if str(word) in embdict.keys() or word_dict[str(word)][1] >= 3:
                sentence.append(word_dict[str(word)][0])
            if len(sentence) == maxlen:
                break
        review.append(sentence + [0] * (maxlen - len(sentence)))
        if len(review) == 150:
            break
    car_train_data.append(review + [[0] * maxlen] * (150 - len(review)))

car_test_data = []
for i in car_test_reviews:
    review = []
    for sent in i:
        sentence = []
        for word in sent:
            if str(word) in embdict.keys() or word_dict[str(word)][1] >= 3:
                sentence.append(word_dict[str(word)][0])
            if len(sentence) == maxlen:
                break
        review.append(sentence + [0] * (maxlen - len(sentence)))
        if len(review) == 150:
            break
    car_test_data.append(review + [[0] * maxlen] * (150 - len(review)))

indices = np.arange(len(car_train_reviews))
np.random.shuffle(indices)
data = np.array(car_train_data)[indices]
labels = np.array(to_categorical(car_train_labels))[indices]

nb_validation_samples = int(0.2 * len(car_train_reviews))
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

x_test = np.array(car_test_data)
y_test = np.array(to_categorical(car_test_labels))

np.save("./data/train/{}_x_train.npy".format(categ), x_train)
np.save("./data/train/{}_x_val.npy".format(categ), x_val)
np.save("./data/train/{}_y_train.npy".format(categ), y_train)
np.save("./data/train/{}_y_val.npy".format(categ), y_val)
np.save("./data/test/{}_x_test.npy".format(categ), x_test)
np.save("./data/test/{}_y_test.npy".format(categ), y_test)
