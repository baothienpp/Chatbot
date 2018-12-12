import tensorflow as tf
import os
import numpy as np
import argparse


class LanguageModel():

    def __init__(self, data_dir, epoch, batch_size, out_dir):
        self.data_dir = data_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.out_dir = out_dir

        self.X, self.Y, self.ix_to_char, self.char_to_ix, self.VOCAB_SIZE = self._build_dataset(self.data_dir,
                                                                                                seq_length=50)
        self.model = self._create_model()

    def _build_dataset(self, train_data, seq_length):
        X = []
        Y = []

        for file in os.listdir(train_data):
            if file.endswith(".txt"):
                with open(os.path.join(train_data, file), encoding='utf-8') as file:
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

    def train(self):
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")
        checkpoint = tf.keras.callbacks.ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='train_loss',
                                                        save_best_only=True,
                                                        mode='auto')

        self.model.fit(self.X, self.Y, batch_size=self.batch_size, verbose=1, nb_epoch=self.epoch,
                       callbacks=[checkpoint])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default='dataset/', help='Set path to dataset folder')
    parser.add_argument('--epoch', default=200, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('-o', '--output', default='model/', help='Model output path')
    args = parser.parse_args()

    model = LanguageModel(args.data, args.epoch, args.batch_size, args.output)
    model.train()
