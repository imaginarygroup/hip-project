import os
from random import random

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
# Generate dummy data
import numpy as np
# For custom metrics
import keras.backend as K
import scipy.io

class HipJointV1(object):

    MINIMUM_FEATURE_POINT_COUNT = 50
    FIT_RATE = 0.75
    EPOCH_TRIES = 1200

    def __init__(self, insize, outsize):
        self.input_size = insize
        self.output_size = outsize
        self.all_maps = {}
        self.all_labels = {}
        self.all_names = []

        # Randomly selected
        self.fit_names = []
        self.predict_names = []

    def load_dat(self):
        """
        Load
        :return:
        """

        pass

    def initialize(self):
        """
        Initialize the model
        :return:
        """
        model = Sequential()
        model.add(Dense(self.input_size, input_dim=self.input_size))
        model.add(Activation('tanh'))
        model.add(BatchNormalization(momentum=0.99))
        model.add(Dense(int(self.input_size / 2), input_shape=(self.input_size,)))
        model.add(Activation('tanh'))
        model.add(Dropout(rate=0.1))
        model.add(Dense(int(self.input_size / 20)))
        model.add(Activation('tanh'))
        model.add(Dense(self.output_size))
        model.add(Activation('softmax'))
        model.compile(optimizer='rmsprop',
                      loss='mse')

        return model

    def fit_model(self, model, data, labels, epoch_times=10):
        # Train the model, iterating on the data in batches of 32 samples
        model.fit(data, labels, epochs=epoch_times, batch_size=32)


    def mean_pred(y_true, y_pred):
        return K.mean(y_pred)

    def compile_with_specific_loss(self, model, mean_pred):
        model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', mean_pred])

    def random_data(self):
        data = np.random.random((1000, self.input_size))
        labels = np.random.randint(2, size=(1000, self.output_size))
        return {"data": data, "labels": labels}

    def predict(self, new_rand_data, sample_size):
        sample_size = new_rand_data.shape[0]
        if len(new_rand_data.shape) < 2 \
                or new_rand_data.shape[0] != sample_size \
                or new_rand_data.shape[1] != self.input_size:
            print("New data to be predicted must have shape [SAMPLE_SIZE(%d), INPUT_SIZE(%d)]" %
                  (sample_size, self.input_size))
            return None

        result = model.predict(new_rand_data)
        print ("Prediction result in shape : ", result.shape)
        return result

    def prediction_data(self):
        new_data = []
        for i in range(0, self.input_size):
            new_data.append(i)
        return np.reshape(np.array(new_data), [1, self.input_size])

    def load_from_all_mat_files(self, root_path):
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.endswith("1.1.mat"):
                    self.deal_single_image(file, root, "1.1")
                if file.endswith("1.2.mat"):
                    self.deal_single_image(file, root, "1.2")

        ## All data is in all_maps
        ## Add to name list
        for k in self.all_maps.keys():
            self.all_names.append(k)

    def load_labels_from_csv(self, file_path):
        import csv
        with open(file_path, newline='') as csvfile:
            all_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in all_reader:
                name = row[0]
                name1 = name + "1.1"
                name2 = name + "1.2"
                self.all_labels[name1] = [float(row[1]), float(row[2]), float(row[3])]
                self.all_labels[name2] = [float(row[4]), float(row[5]), float(row[6])]
        pass

    def deal_single_image(self, file, root, suffix):
        name = os.path.split(root)[1] + suffix
        whole_path = os.path.join(root, file)
        print("Found one in ", suffix, " : ", whole_path)
        mat = scipy.io.loadmat(whole_path)
        res = mat.get("f").T
        dtype = [('x', float), ('y', float), ('radius', float), ('direction', float)]
        a = res.view(dtype)  # create a structured array
        a = a.reshape((a.shape[0], ))
        sort_res = np.sort(a, order='radius')
        if sort_res.shape[0] <= HipJointV1.MINIMUM_FEATURE_POINT_COUNT:
            return
        reviewed_res = sort_res.view(res.dtype).reshape((res.shape[0], 4))
        slice_res = reviewed_res[-(HipJointV1.MINIMUM_FEATURE_POINT_COUNT):]
        print("Name : ", name, "Mat Shape :", slice_res.shape)
        reshape_res = slice_res.reshape((HipJointV1.MINIMUM_FEATURE_POINT_COUNT * 4, ))
        print("Reshaped shape : ", reshape_res.shape)
        self.all_maps[name] = reshape_res

    def get_fit_data(self):
        cur_list = []

        for one in self.all_names:
            if self.all_maps.get(one) is None or self.all_labels.get(one) is None:
                continue
            if random() < HipJointV1.FIT_RATE:
                self.fit_names.append(one)
            else:
                self.predict_names.append(one)

        for one in self.fit_names:
            cur_list.append(self.all_maps.get(one))

        cur_array = np.stack(cur_list, axis=0)
        return cur_array

    def get_fit_labels(self):
        cur_list = []

        for one in self.fit_names:
            cur_list.append(np.array(self.all_labels.get(one)))

        cur_array = np.stack(cur_list, axis=0)
        return cur_array


    def get_predict_labels(self):
        cur_list = []

        for one in self.predict_names:
            cur_list.append(np.array(self.all_labels.get(one)))

        cur_array = np.stack(cur_list, axis=0)
        return cur_array

    def get_predict_data(self):
        cur_list = []

        for one in self.predict_names:
            cur_list.append(self.all_maps.get(one))

        cur_array = np.stack(cur_list, axis=0)
        return cur_array


if __name__ == "__main__":
    hv1 = HipJointV1(HipJointV1.MINIMUM_FEATURE_POINT_COUNT * 4, 3)
    ## If model exists, load model
    model = hv1.initialize()

    ## 1. Open mat files and read the 4 * (50-150)
    ## 1.1. Sort the mat file
    ## 1.2. Stack onto
    hv1.load_from_all_mat_files("/Users/xc5/PycharmProjects/hipjoint/s1")
    ## 2. Open the CSV label file(with given data)
    hv1.load_labels_from_csv("/Users/xc5/PycharmProjects/hipjoint/hipjoint2.csv")

    ## 3. Random split data into 2 folds
    fit_data = hv1.get_fit_data()
    fit_labels = hv1.get_fit_labels()
    random_data = hv1.get_predict_data()
    random_labels = hv1.get_predict_labels()

    ## 4. Fit
    hv1.fit_model(model, fit_data, fit_labels, HipJointV1.EPOCH_TRIES)
    # hv1.fit_model(model, rand_data.get("data"), rand_data.get("labels"))

    ## 5. Predict, calculate loss
    ## MSE,
    print ("Predicting ... ")
    print(random_data.shape)

    random_predict_result = hv1.predict(random_data, len(random_labels))
    print(random_predict_result)
    print(random_labels)

    mse = (np.square(random_predict_result - random_labels)).mean(axis=None)

    print ("Global MSE = ", mse)

    ## Saves the model
    ## hv1.dump_model

    ## Sample size as 1, due to prediction_data only generate one example
