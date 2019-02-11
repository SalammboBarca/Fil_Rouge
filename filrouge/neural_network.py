import configparser
import pandas as pd
import itertools
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD,  Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from filrouge.vectorize_comments import VectorTfidf


class NeuralNetwork:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('../config.cfg')
        self.train_df = pd.read_csv(config['FILES']['TRAIN'])
        self.test_df = pd.read_csv(config['FILES']['TEST'])
        self.cols_name = ['obscene', 'insult', 'toxic', 'severe_toxic', 'identity_hate', 'threat']
        self.test_label = pd.read_csv(config['FILES']['LABEL'])
        vect = VectorTfidf()
        self.train_vect, self.test_vect = vect.vectorize()
        # print(self.train_vect.shape, self.test_vect.shape)

    def transform_label(self):
        results = list(self.cols_name)
        ynew = []
        for x in range(2, len(self.cols_name) + 1):
            results.extend(itertools.combinations(self.cols_name, x))
        sample_size = len(self.train_vect.shape[0])
        for sample in range(0, sample_size):
            y = []
            for label in enumerate(self.cols_name):
                if self.train_df[label] != 0:
                    y.append(label)
            if len(y) == 1:
                ynew.append(self.cols_name.index(str(y[0])))
            else:
                ynew.append(results.index())

    def createModel(self, epochs, batch_size):
        model = Sequential()
        nhu = 800
        model.add(Dense(units=nhu, input_shape=(5000,)))
        model.add(Activation("relu"))
        # ajout de la couche de sortie
        model.add(Dense(units=5))
        model.add(Activation("softmax"))
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        # early_stop1 = EarlyStopping(monitor="val_loss", min_delta=0, patience=2, verbose=0)
        early_stop2 = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=1,
                                      save_best_only=False, save_weights_only=False, mode='auto', period=1)
        model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=adam)
        # hist = model.fit(Xa, Ya, epochs=epochs, batch_size=batch_size)
        # hist = model.fit(Xa, Ya, validation_split=0.25, epochs=epochs, batch_size=batch_size)
        model.fit(self.train, self.ya, validation_split=0.25, epochs=epochs, batch_size=batch_size, callbacks=[early_stop2])
        # plt.figure(), plt.subplot(1, 2, 1)
        # plt.plot(range(1, epochs + 1), hist.history["loss"], color="blue", label="Train loss")
        # plt.plot(range(1, epochs + 1), hist.history["val_loss"], color="red", label="Val loss")
        # plt.xlabel("Epochs"), plt.title("Loss function"), plt.legend(loc="best")
        # plt.subplot(1, 2, 2)
        # plt.plot(range(1, epochs + 1), hist.history["acc"], color="blue",
        #          label="Train acc")
        # plt.plot(range(1, epochs + 1), hist.history["val_acc"], color="red",
        #          label="Val acc")
        # plt.xlabel("Epochs"), plt.title("Accuracy"), plt.legend(loc="best")
        # plt.show()
        score_test = model.evaluate(self.test, self.yt)
        print("\nLoss (test): %.3f" % score_test[0])
        print("Taux classif (test): %.3f" % score_test[1])


if __name__ == '__main__':
    neurone = NeuralNetwork()
    neurone.transform_label()
