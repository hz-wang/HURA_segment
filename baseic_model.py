import os
import numpy as np
import codecs
from sklearn.metrics import classification_report, accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, concatenate, Convolution1D, Lambda, GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, LSTM, Bidirectional, MaxPooling1D, AveragePooling1D, Input
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.models import KeyedVectors
from keras.utils import plot_model
# from nltk import Tokenizer
from keras import optimizers

TYPES ='car'
embedding_matrix = np.load('./data/{}_embedding_matrix.npy'.format(TYPES))
MAX_SEQUENCE_LENGTH = 200
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
CONCEPT_EMBEDDING_DIM = 20
NB_CONCEPT = 10
nb_words = embedding_matrix.shape[0]

print('Build model...')
#######
# LSTM
#######
# model = Sequential()
# model.add(Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
# model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
# # model.add(Bidirectional(LSTM(128)))
# model.add(Dense(1, activation='sigmoid'))

########
# CNN
########
# nb_filter = 200
# filter_length = 3
# hidden_dim = 100
# model = Sequential()
# model.add(Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
# model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length,  padding='valid', activation='relu', strides=1)) # padding='valid',
# model.add(GlobalMaxPooling1D())
# # model.add(Dense(hidden_dim, activation='relu'))
# # model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))

########
# concept_CNN
########
embedding_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=True)
# concept_embedding_layer = Embedding(len(concept2id) + 1, CONCEPT_EMBEDDING_DIM, weights=[concept_embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=True)

sentence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name="word_input")
# concept_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='concept_input') # single
# concept_input = Input(shape=(MAX_SEQUENCE_LENGTH, NB_CONCEPT), dtype='int32', name='concept_input') # multi

embedded_sequences = embedding_layer(sentence_input)
# concept_embedded_sequences = concept_embedding_layer(concept_input)

# __import__("ipdb").set_trace()
# concept_embedded_sequences = Lambda(lambda x: K.mean(x, axis=2), output_shape=lambda s: (s[0], s[1], s[2]))(concept_embedded_sequences)

# embedded_sequences = concatenate([embedded_sequences, concept_embedded_sequences], axis=-1)
l_cnn = Convolution1D(filters=200, kernel_size=3,  padding='same', activation='relu', strides=1)(embedded_sequences)
l_cnn = GlobalMaxPooling1D()(l_cnn)
# l_cnn = Dropout(0.2)(l_cnn)
# l_lstm = Bidirectional(LSTM(100))(l_cnn)
preds = Dense(1, activation='sigmoid')(l_cnn)
model = Model(sentence_input, preds)
# model = Model([sentence_input, concept_input], preds)

###########
# CNN_LSTM
###########

# model = Sequential()
# model.add(Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=True))
# model.add(Convolution1D(nb_filter=100, filter_length=3,  border_mode='valid', activation='relu', strides=1)) # padding='valid',
# model.add(MaxPooling1D(4))
# model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))

##################
# LSTM MAX POOLING
##################

# embedding_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
# data_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(data_input)
# lstm_layer = Bidirectional(LSTM(200, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
# pool_layer = GlobalMaxPooling1D()
# state_LSTM = lstm_layer(embedded_sequences)
# state_LSTM = pool_layer(state_LSTM)
# merged = Dropout(0.2)(state_LSTM)
# merged = Dense(100, activation='relu')(merged)
# merged = Dropout(0.2)(merged)
# preds = Dense(1, activation='sigmoid')(merged)
# model = Model(inputs=[data_input], outputs=preds)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# plot_model(model, to_file='model.png')

print('Train...')
batch_size = 100
def train_model(model, x_train, y_train, x_val, y_val, path):
    print("model fitting - Hierachical attention network")
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    bst_model_path = path+'.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=1, epochs=3, batch_size=32, callbacks=[early_stopping, model_checkpoint])


x_train = np.load('./data/basic/train/{}_x_train.npy'.format(TYPES))
y_train = np.load('./data/basic/train/{}_y_train.npy'.format(TYPES))
x_val = np.load('./data/basic/train/{}_x_val.npy'.format(TYPES))
y_val = np.load('./data/basic/train/{}_y_val.npy'.format(TYPES))
x_test = np.load('./data/basic/test/{}_x_test.npy'.format(TYPES))
y_test = np.load('./data/basic/test/{}_y_test.npy'.format(TYPES))
#
train_model(model, x_train, y_train, x_val, y_val, './code/model/CNN' )
model = load_model('./code/model/CNN.h5')

y_pred = model.predict(x_test)
y_pred = y_pred.round()
y_true = y_test
acc = accuracy_score(y_true, y_pred)
print('acc: ', acc)
cr = classification_report(y_true, y_pred, labels=[0, 1], target_names=['pos', 'neg'], digits=4)
print(cr)