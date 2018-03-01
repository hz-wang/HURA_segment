import re
import codecs
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

import numpy as np

MAX_SEQUENCE_LENGTH = 200
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
CONCEPT_EMBEDDING_DIM = 20
NB_CONCEPT = 10

CAR_TRAIN_PATH = './data/pre/onlytrainnoquery/car_train.txt'
CAR_TEST_PATH = './data/pre/onlytrainnoquery/car_test.txt'
PET_TRAIN_PATH = './data/pre/onlytrainnoquery/pet_train.txt'
PET_TEST_PATH = './data/pre/onlytrainnoquery/pet_test.txt'


def load_data(path):
    texts = []
    labels = []
    file = codecs.open(path, "r", encoding='utf-8', errors='ignore')
    for line in file.readlines():
        terms = line.split('\t')
        if (len(terms) < 3):
            continue
        labels.append(int(terms[0]))
        texts.append(terms[1].strip() + "\t" + terms[2].strip())
    file.close()
    return labels, texts

def bulid_data(texts):
    # print(labels)
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index

    sequences = tokenizer.texts_to_sequences(texts)

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print(data[5])
    nb_words = min(MAX_NB_WORDS, len(tokenizer.word_index)) + 1
    return data, word_index, nb_words

def load_embedding(word_index, nb_words):
    GLOVE_DIR = "./code"
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.random.random((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and i < nb_words:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix

def main(categ):

    train_path = './data/{}_train.txt'.format(categ)
    test_path = './data/{}_test.txt'.format(categ)

    labels, texts = load_data(train_path)
    print("Number of users:", len(labels))
    data, word_index, nb_words = bulid_data(texts)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = np.asarray(data)
    labels = np.asarray(labels)
    x_data = data[indices]
    y_data = labels[indices]
    training_num = int(len(y_data) * 0.8)
    x_train = x_data[:training_num]
    y_train = y_data[:training_num]
    x_val = x_data[training_num:]
    y_val = y_data[training_num:]

    embedding_matrix = load_embedding(word_index, nb_words)

    print('Train: Number of positive and negative reviews in traing and validation set')
    print(y_train.sum(axis=0))
    print(y_val.sum(axis=0))
    np.save("./data/basic/train/{}_x_train.npy".format(categ), x_train)
    np.save("./data/basic/train/{}_x_val.npy".format(categ), x_val)
    np.save("./data/basic/train/{}_y_train.npy".format(categ), y_train)
    np.save("./data/basic/train/{}_y_val.npy".format(categ), y_val)
    np.save("./data/basic/{}_embedding_matrix.npy".format(categ), embedding_matrix)


    labels, texts = load_data(test_path)
    print("Test: Number of users:", len(labels))
    data, word_index, nb_words = bulid_data(texts)
    x_test = data
    y_test = labels
    np.save("./data/basic/test/{}_x_test.npy".format(categ), x_test)
    np.save("./data/basic/test/{}_y_test.npy".format(categ), y_test)


if __name__ == '__main__':
    main('pet')
    main('car')



