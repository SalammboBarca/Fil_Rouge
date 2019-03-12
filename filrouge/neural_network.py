import configparser
import pandas as pd
import itertools
import matplotlib.pyplot as plt
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

    def transform_label(self, df)->list:
        """

        :param df: dataframe containing multi labels
        :return: new label vector with classes representing combination of labels
        """
        results = [str(classe) for classe in range(1, len(self.cols_name) + 1)]
        classes = [str(classe) for classe in range(1, len(self.cols_name) + 1)]
        ynew = []
        for x in range(2, len(self.cols_name) + 1):
            results.extend([''.join(comb) for comb in itertools.combinations(classes, x)])
        sample_size = df.shape[0]
        for sample in range(0, sample_size):
            y = ''
            for idx, label in enumerate(self.cols_name):
                if df[label][sample] != 0:
                    y = ''.join([y, str(idx + 1)])
            if y:
                ynew.append(y)
            else:
                ynew.append('0')
        print(len(ynew))
        for classe in results:
            print('On a {} commentaires dans la classe {}'.format(ynew.count(classe), classe))
        return ynew

    def create_model(self, ya, yt, epochs, batch_size):
        model = Sequential()
        nhu = 800
        model.add(Dense(units=nhu, input_shape=(5000,)))
        model.add(Activation("relu"))
        # ajout de la couche de sortie
        model.add(Dense(units=63))
        model.add(Activation("softmax"))
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        # early_stop1 = EarlyStopping(monitor="val_loss", min_delta=0, patience=2, verbose=0)
        # early_stop2 = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=1,
        # save_best_only=False, save_weights_only=False, mode='auto', period=1)
        model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=adam)
        hist = model.fit(self.train_vect, ya, epochs=epochs, batch_size=batch_size)
        # hist = model.fit(Xa, Ya, validation_split=0.25, epochs=epochs, batch_size=batch_size)
        # model.fit(self.train, self.ya, validation_split=0.25, epochs=epochs, batch_size=batch_size,
        # callbacks=[early_stop2])
        plt.figure(), plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), hist.history["loss"], color="blue", label="Train loss")
        plt.plot(range(1, epochs + 1), hist.history["val_loss"], color="red", label="Val loss")
        plt.xlabel("Epochs"), plt.title("Loss function"), plt.legend(loc="best")
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), hist.history["acc"], color="blue",
                 label="Train acc")
        plt.plot(range(1, epochs + 1), hist.history["val_acc"], color="red",
                 label="Val acc")
        plt.xlabel("Epochs"), plt.title("Accuracy"), plt.legend(loc="best")
        plt.show()
        score_test = model.evaluate(self.test_vect, yt)
        print("\nLoss (test): %.3f" % score_test[0])
        print("Taux classif (test): %.3f" % score_test[1])

    def main(self):
        labels_train = self.transform_label(self.train_df)
        labels_test = self.transform_label(self.test_label)
        self.create_model(labels_train, labels_test, 20, 32)


if __name__ == '__main__':
    neurone = NeuralNetwork()
    neurone.main()
