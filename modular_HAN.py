import pandas as pd
from collections import defaultdict
import re
import codecs

from bs4 import BeautifulSoup

import sys
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding, concatenate
from keras.layers import Dense, Input, Flatten, average,Lambda
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers #keras2
from keras.utils import plot_model
import numpy as np
from sklearn.metrics import classification_report

embedding_matrix = np.load('./data/car_embedding_matrix.npy')
MAX_SENT_LENGTH = 80
MAX_SENTS = 150
MAX_NB_WORDS = 20000
WORD_INDEX = embedding_matrix.shape[0]
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

class AttLayer(Layer):
    def __init__(self, **kwargs):
        # self.init = initializations.get('normal')#keras1.2.2
        self.init = initializers.get('normal')

        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], 1)))
        # self.W = self.init((input_shape[-1],))
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        # eij = K.tanh(K.dot(x, self.W))
        print(x.shape)
        print(self.W.shape)
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        print(ai.shape)
        # weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')
        weights = ai / K.expand_dims(K.sum(ai, axis=1), 1)
        print('weights', weights.shape)
        # weighted_input = x * weights.dimshuffle(0, 1, 'x')
        weighted_input = x * weights

        # return weighted_input.sum(axis=1)
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

def build_model():
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_cnn = Convolution1D(nb_filter=200, filter_length=3,  padding='valid', activation='relu', strides=1)(embedded_sequences)
    l_cnn = MaxPooling1D(4)(l_cnn)
    #l_cnn = Dropout(0.2)(l_cnn)
    l_lstm = Bidirectional(LSTM(100, return_sequences=True))(l_cnn)
    l_lstm = Dropout(0.2)(l_lstm)
    l_dense = TimeDistributed(Dense(200))(l_lstm)
    l_att = AttLayer()(l_dense)
    sentEncoder = Model(sentence_input, l_att)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    print(review_encoder.shape)
    #l_lstm_sent = Bidirectional(LSTM(100, return_sequences=True))(review_encoder)
    l_cnn_sent = Convolution1D(nb_filter=200, filter_length=3, padding='valid', activation='relu', strides=1)(review_encoder)
    l_cnn_sent = MaxPooling1D(4)(l_cnn_sent)
    l_cnn_sent = Dropout(0.2)(l_cnn_sent)
    #l_embed_sent = Dropout(0.5)(l_cnn_sent)
    #l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
    l_att_sent = AttLayer()(l_cnn_sent)
    preds = Dense(2, activation='sigmoid')(l_att_sent)
    model = Model(review_input, preds)


    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model

def CL(use_attw=True, use_atts=True):
    model_name = 'CNNW-LSTMS'
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_cnn = Convolution1D(nb_filter=200, filter_length=3,  padding='same', activation='relu', strides=1)(embedded_sequences)
    if use_attw:
        l_att = AttLayer()(l_cnn)
    else:
        l_att = Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s: (s[0], s[2]))(l_cnn)
    sentEncoder = Model(sentence_input, l_att)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(LSTM(100, return_sequences=True))(review_encoder)
    l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
    if use_atts:
        l_att_sent = AttLayer()(l_dense_sent)
    else:
        l_att_sent = Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s: (s[0], s[2]))(l_dense_sent)
    preds = Dense(2, activation='sigmoid')(l_att_sent)
    model = Model(review_input, preds)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model, model_name

def LL(use_attw=True, use_atts=True):
    model_name = 'LSTMW-LSTMS'
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
    # l_lstm = Dropout(0.2)(l_lstm)
    l_dense = TimeDistributed(Dense(200))(l_lstm)
    if use_attw:
        l_att = AttLayer()(l_dense)
    else:
        l_att = Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s: (s[0], s[2]))(l_dense)
    sentEncoder = Model(sentence_input, l_att)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(LSTM(100, return_sequences=True))(review_encoder)
    l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
    if use_atts:
        l_att_sent = AttLayer()(l_dense_sent)
    else:
        l_att_sent = Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s: (s[0], s[2]))(l_dense_sent)
    preds = Dense(2, activation='sigmoid')(l_att_sent)
    model = Model(review_input, preds)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model, model_name

def CLL(use_attw=True, use_atts=True):
    model_name = 'CNNW-LSTMW-LSTMS'
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_cnn = Convolution1D(nb_filter=200, filter_length=3,  padding='same', activation='relu', strides=1)(embedded_sequences)
    l_lstm = Bidirectional(LSTM(100, return_sequences=True))(l_cnn)
    l_dense = TimeDistributed(Dense(200))(l_lstm)
    if use_attw:
        l_att = AttLayer()(l_dense)
    else:
        l_att = Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s: (s[0], s[2]))(l_dense)
    sentEncoder = Model(sentence_input, l_att)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(LSTM(100, return_sequences=True))(review_encoder)
    l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
    if use_atts:
        l_att_sent = AttLayer()(l_dense_sent)
    else:
        l_att_sent = Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s: (s[0], s[2]))(l_dense_sent)
    preds = Dense(2, activation='sigmoid')(l_att_sent)
    model = Model(review_input, preds)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model, model_name

def CC(use_attw=True, use_atts=True):
    model_name = 'CNNW-CNNS'
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_cnn = Convolution1D(nb_filter=200, filter_length=3,  padding='same', activation='relu', strides=1)(embedded_sequences)
    if use_attw:
        l_att = AttLayer()(l_cnn)
    else:
        l_att = Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s: (s[0], s[2]))(l_cnn)
    sentEncoder = Model(sentence_input, l_att)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_cnn_sent = Convolution1D(nb_filter=200, filter_length=3, padding='same', activation='relu', strides=1)(review_encoder)
    if use_atts:
        l_att = AttLayer()(l_cnn_sent)
    else:
        l_att = Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s: (s[0], s[2]))(l_cnn_sent)
    preds = Dense(2, activation='sigmoid')(l_att)
    model = Model(review_input, preds)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model, model_name

def LC(use_attw=True, use_atts=True):
    model_name = 'LSTMW-CNNS'
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
    l_dense = TimeDistributed(Dense(200))(l_lstm)
    if use_attw:
        l_att = AttLayer()(l_dense)
    else:
        l_att = Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s: (s[0], s[2]))(l_dense)
    sentEncoder = Model(sentence_input, l_att)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_cnn_sent = Convolution1D(nb_filter=200, filter_length=3, padding='same', activation='relu', strides=1)(review_encoder)
    if use_atts:
        l_att = AttLayer()(l_cnn_sent)
    else:
        l_att = Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s: (s[0], s[2]))(l_cnn_sent)
    preds = Dense(2, activation='sigmoid')(l_att)
    model = Model(review_input, preds)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',  metrics=['acc'])
    return model, model_name

def CLC(use_attw=True, use_atts=True):
    model_name = 'CNNW-LSTMW-CNNS'
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_cnn = Convolution1D(nb_filter=200, filter_length=3,  padding='same', activation='relu', strides=1)(embedded_sequences)
    l_lstm = Bidirectional(LSTM(100, return_sequences=True))(l_cnn)
    l_dense = TimeDistributed(Dense(200), name='Dense')(l_lstm)
    if use_attw:
        l_att = AttLayer()(l_dense)
    else:
        l_att = Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s: (s[0], s[2]))(l_dense)
    sentEncoder = Model(sentence_input, l_att)
    
    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder, name='sentEncoder')(review_input)
    l_cnn_sent = Convolution1D(nb_filter=200, filter_length=3, padding='same', activation='relu', strides=1)(review_encoder)
    if use_atts:
        l_att = AttLayer()(l_cnn_sent)
    else:
        l_att = Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s: (s[0], s[2]))(l_cnn_sent)
    preds = Dense(2, activation='sigmoid')(l_att)
    model = Model(review_input, preds)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model, model_name

def train_model(model, x_train, y_train, x_val, y_val, count, path, model_name):
    print("model fitting - Hierachical attention network")
    early_stopping = EarlyStopping(monitor='val_acc', patience=3)
    bst_model_path = path+'.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=1, epochs=10, batch_size=32, callbacks=[early_stopping, model_checkpoint])

    score, acc = model.evaluate(x_val, y_val, batch_size=100)

    print('This is the', count, ' time for model', model_name)
    print('Test score:', score)
    print('Test accuracy:', acc)
from nltk import tokenize

def train_test(model, name):
    x_train = np.load('./data/train/car_x_train.npy')
    y_train = np.load('./data/train/car_y_train.npy')
    x_val = np.load('./data/train/car_x_val.npy')
    y_val = np.load('./data/train/car_y_val.npy')
    x_test = np.load('./data/test/car_x_test.npy')
    y_test = np.load('./data/test/car_y_test.npy')

    train_model(model, x_train, y_train, x_val, y_val, count, './code/model/{}'.format(name), name)
    model = load_model('./code/model/{}.h5'.format(name), custom_objects={'AttLayer': AttLayer})
    score, acc = model.evaluate(x_test, y_test, batch_size=100)
    y_pred = model.predict(x_test)
    print('No: ', count, 'acc: ', acc)
    cr = classification_report(y_test, y_pred, labels=[0, 1], target_names=['pos', 'neg'], digits=3)
    print(cr)
    writer = codecs.open('./data/report/car.txt', "a", encoding='utf-8', errors='ignore')
    writer.write('model: ', name, 'No: ', count, 'acc: ', acc, '\n')
    writer.write(cr, '\n')

if __name__ == '__main__':
    embedding_layer = Embedding(WORD_INDEX, EMBEDDING_DIM, weights=[embedding_matrix],
                                input_length=MAX_SENT_LENGTH, trainable=True)

    # for count in range(5):
    #     model, model_name = CL(use_attw=False, use_atts=False)
    #     train_test(model, 'CL')
    # for count in range(5):
    #     model, model_name = LL(use_attw=False, use_atts=False)
    #     train_test(model, 'LL')
    # for count in range(5):
    #     model, model_name = CLL(use_attw=False, use_atts=False)
    #     train_test(model, 'CLL')
    # for count in range(5):
    #     model, model_name = CC(use_attw=False, use_atts=False)
    #     train_test(model, 'CC')
    # for count in range(5):
    #     model, model_name = LC(use_attw=False, use_atts=False)
    #     train_test(model, 'LC')
    # for count in range(5):
    #     model, model_name = CLC(use_attw=False, use_atts=False)
    #     train_test(model, 'CLC')
    # for count in range(5):
    #     model, model_name = CLC(use_attw=True, use_atts=False)
    #     train_test(model, 'CALC')
    # for count in range(5):
    #     model, model_name = CLC(use_attw=False, use_atts=True)
    #     train_test(model, 'CLCA')
    for count in range(1):
        # model, model_name = CLC(use_attw=True, use_atts=True)
        # train_test(model, 'CLACA')
        model = load_model('./code/model/CLACA.h5', custom_objects={'AttLayer': AttLayer})
        x_test = np.load('./data/test/pet_x_test.npy')
        print('load x finish')
        y_test = np.load('./data/test/pet_y_test.npy')
        print('load y finish')
        y_pred = model.predict(x_test)
        print('pred finish')
        cr = classification_report(y_test, y_pred, labels=[0, 1], target_names=['pos', 'neg'], digits=3)
        print(cr)