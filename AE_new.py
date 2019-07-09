from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras import optimizers
import model
import Files
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

class AutoEncoder():
    def __init__(self, structure=[1,2,1], title=""):
        self.name = title
        self.init_layers(structure)
        self.init_model()

    def init_layers(self, structure):
        start_encode = 1
        end_encode = int(len(structure) - (len(structure) - 1) / 2)
        start_decode = int(end_encode)
        end_decode = int(len(structure))

        self.input = Input(shape=(structure[0],))
        for i in range(start_encode, end_encode):
            if i == start_encode:
                self.encoded = Dense(units=structure[i], activation='relu')(self.input)
            else:
                self.encoded = Dense(units=structure[i], activation='relu')(self.encoded)
        for i in range(start_decode, end_decode):
            if i == start_decode:
                self.decoded = Dense(units=structure[i], activation='relu')(self.encoded)
            elif i > start_decode and i < end_decode:
                self.decoded = Dense(units=structure[i], activation='relu')(self.decoded)
            elif i == end_decode - 1:
                self.decoded = Dense(units=structure[i], activation='sigmoid')(self.decoded)

    def init_model(self):
        self.autoencoder = Model(self.input, self.decoded)

    def print_model_summary(self):
        print(self.autoencoder.summary())

    def compile_model(self, optimizer, loss='mean_squared_error', metrics=['mean_squared_error', "accuracy"]):
        self.autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def prepare_data(self, training_length, data_len):
        logging.info("prepare_data - reading data...")
        dataset = Files.read_data(data_len)
        dataset_train = dataset[:training_length]
        dataset_test = dataset[training_length:]
        logging.info("prepare_data - read data...")
        return dataset_train, dataset_test

    def train_model(self, epochs, training_data_len, batch_size, data_len):
        optimiser = optimizers.Adam(lr=0.001)
        self.compile_model(optimiser)
        dataset_train, dataset_test = self.prepare_data(training_data_len, data_len)
        cb = EarlyStopping(monitor="mean_squared_error", min_delta=0.1, patience=30)
        self.history = self.autoencoder.fit(dataset_train, dataset_train, epochs=epochs, batch_size=256, shuffle=False,
                                       validation_data=(dataset_test, dataset_test), callbacks=[cb], verbose=0)

        self.save_model("model_{}".format(self.name), self.autoencoder)
        self.save_training_results(self.history.history, "Training_Results/"+self.name+"_tests.csv")
        #self.test_model(dataset_test)
        #self.plot_training_data("")
        return self.history

    def encode(self, input):
        logging.info("encode - encoding input..")
        output = self.autoencoder.predict(input)
        return output

    def test_model(self, test):
        data = {}
        logging.info("test_model - testing model")
        predict = self.autoencoder.predict(test)
        print(np.shape(predict))
        predict_labels = []
        for item in model.MEASUREMENT_NODES_2:
            data[item] = []
            data[item+"_predict"] = []
            data[item+"_SE"] = []
        data["MSE"]= []
        sum_difference = 0
        for i in range(0, len(predict)):
            for item in model.MEASUREMENT_NODES_2:
                data[item+"_SE"].append((test[i][model.MEASUREMENT_NODES_2.index(item)]-predict[i][model.MEASUREMENT_NODES_2.index(item)])**2)
                data[item].append(test[i][model.MEASUREMENT_NODES_2.index(item)])
                data[item+"_predict"].append(predict[i][model.MEASUREMENT_NODES_2.index(item)])
                sum_difference += test[i][model.MEASUREMENT_NODES_2.index(item)] - predict[i][model.MEASUREMENT_NODES_2.index(item)]
            data["MSE"].append((sum_difference/len(model.MEASUREMENT_NODES_2))**2)
            sum_difference=0

        print(data)
        df = pd.DataFrame(data=data)
        df.to_csv("/Users/Robin/Desktop/ML Project/Code/Test_Results/{}".format(self.name+"_test_results.csv"), index=True, header=True)
        #logging.debug("test_model - true values: {}".format(test[0][0:5]))
        #logging.debug("test_model - pred values: {}".format(predict[0][0:5]))


    def save_model(self, name, model):
        logging.info("save_model - saving model...")
        # serialize model to JSON
        model_json = model.to_json()
        with open("classifiers/"+name + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("classifiers/"+name + "_weights.h5")
        logging.info("save_model - saved model to disk")

    def load_model(self, location):
        logging.info("load_model - reading model...")
        json_file = open(location+"/"+self.name+".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(location+"/"+self.name + "_weights.h5")
        self.autoencoder = loaded_model
        logging.info("load_model - loaded model from disk")


    def save_training_results(self, history, filename):
        history_df = pd.DataFrame(data=history)
        history_df.to_csv(filename)
        return history_df

    def load_training_results(self, filename):
        history_df = pd.read_csv(filename)
        return history_df

    def plot_training_data(self, val):
        # Plot training & validation loss values
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
if __name__ == "__main__":
    AE = AutoEncoder([18, 512, 256, 128, 256, 512, 18], title="simulation_data_model2")
    #AE.train_model(epochs=500, training_data_len=5000000, batch_size=256, data_len=6000000)
    AE.test_model([18, 22, 22, 18, 22, 22,18, 22, 22,18, 22, 22,18, 22, 22,18, 22, 22])
    
