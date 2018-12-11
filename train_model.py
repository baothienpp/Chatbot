import tensorflow as tf
import numpy as np
import argparse


class LanguageModel():

    def __init__(self, train_data):
        self.X, self.Y, self.ix_to_char, self.char_to_ix, self.VOCAB_SIZE = self._build_dataset(train_data,
                                                                                                seq_length=50)

    def _build_dataset(self, train_data, seq_length):
        X = []
        Y = []

        # TODO loop through all file in folder and concate
        with open(train_data, encoding='utf-8') as file:
            data = file.read()
            data = data.lower()

        chars = list(set(data))
        VOCAB_SIZE = len(chars)

        ix_to_char = {ix: char for ix, char in enumerate(chars)}
        char_to_ix = {char: ix for ix, char in enumerate(chars)}

        for idx in range(0, len(data) - seq_length, 1):
            X_seq = data[idx:idx + seq_length]
            Y_seq = data[idx + seq_length]
            X.append([char_to_ix[char] for char in X_seq])
            Y.append(char_to_ix[Y_seq])

        X = np.reshape(X, (len(X), seq_length, 1))
        X = self._get_one_hot(X[:, :, 0], VOCAB_SIZE)
        Y = self._get_one_hot(Y, VOCAB_SIZE)

        return X, Y, ix_to_char, char_to_ix, VOCAB_SIZE

    def _get_one_hot(self, targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(np.array(targets).shape) + [nb_classes])

    def _create_model(self, seq_length=50, x_features=38):
        X = tf.keras.layers.Input(shape=(seq_length, x_features))
        lstm1 = tf.keras.layers.LSTM(256, return_sequences=True)(X)
        lstm1 = tf.keras.layers.Dropout(0.5)(lstm1)
        lstm2 = tf.keras.layers.LSTM(256)(lstm1)
        lstm2 = tf.keras.layers.Dropout(0.5)(lstm2)

        dense = tf.keras.layers.Dense(38, activation='softmax')(lstm2)

        model = tf.keras.models.Model(inputs=X, outputs=dense)

        return model

    def train(self, model):
        model.compile(loss="categorical_crossentropy", optimizer="adam")
        model.fit(self.X, self.Y, batch_size=32, verbose=1, nb_epoch=1)


model_instance = LanguageModel('dataset/shakespeare.txt')
model = model_instance._create_model()
model_instance.train(model)
