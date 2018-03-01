import re
import codecs
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical

import numpy as np

MAX_SENT_LENGTH = 80
MAX_SENTS = 150
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
CAR_TRAIN_PATH = './data/pre/onlytrainnoquery/car_train.txt'
CAR_TEST_PATH = './data/pre/onlytrainnoquery/car_test.txt'
PET_TRAIN_PATH = './data/pre/onlytrainnoquery/pet_train.txt'
PET_TEST_PATH = './data/pre/onlytrainnoquery/pet_test.txt'

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def load_data(path):
    reviews = []
    labels = []
    texts = []
    file = codecs.open(path, "r", encoding='utf-8', errors='ignore')
    for line in file.readlines():
        terms = line.split('\t')
        query = terms[1]
        title = terms[2]
        query = clean_str(query.encode('ascii', 'ignore').decode('utf-8', 'ignore'))
        title = clean_str(title.encode('ascii', 'ignore').decode('utf-8', 'ignore'))
        text = title+"#"+query
        sentences = text.split('#')
        texts.append(text)
        reviews.append(sentences)
        labels.append(int(terms[0]))
    file.close()
    return reviews, labels, texts

def bulid_data(reviews, labels, texts):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    print(len(texts))

    data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

    for i, sentences in enumerate(reviews):
        for j, sent in enumerate(sentences):
            if j < MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                k = 0
                for _, word in enumerate(wordTokens):
                    if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                        data[i, j, k] = tokenizer.word_index[word]
                        k = k + 1

    word_index = tokenizer.word_index
    print('Total %s unique tokens.' % len(word_index))

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    return data, labels, word_index

def load_embedding(word_index):
    GLOVE_DIR = "./code"
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors.' % len(embeddings_index))

    # building Hierachical Attention network
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print(len(word_index))
    return embedding_matrix

def main(categ):

    train_path = './data/{}_train.txt'.format(categ)
    test_path = './data/{}_test.txt'.format(categ)

    reviews, labels, texts = load_data(train_path)
    print("Number of users:", len(labels))
    data, labels, word_index = bulid_data(reviews, labels, texts)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    embedding_matrix = load_embedding(word_index)
    print('Train: Number of positive and negative reviews in traing and validation set')
    print(y_train.sum(axis=0))
    print(y_val.sum(axis=0))
    np.save("./data/train/{}_x_train.npy".format(categ), x_train)
    np.save("./data/train/{}_x_val.npy".format(categ), x_val)
    np.save("./data/train/{}_y_train.npy".format(categ), y_train)
    np.save("./data/train/{}_y_val.npy".format(categ), y_val)
    np.save("./data/{}_embedding_matrix.npy".format(categ), embedding_matrix)


    reviews, labels, texts = load_data(test_path)
    print("Test: Number of users:", len(labels))
    data, labels, word_index = bulid_data(reviews, labels, texts)
    x_test = data
    y_test = labels
    np.save("./data/test/{}_x_test.npy".format(categ), x_test)
    np.save("./data/test/{}_y_test.npy".format(categ), y_test)


if __name__ == '__main__':
    main('pet')
    main('car')



