import keras as K

from keras import Sequential
from keras.optimizers import adam, rmsprop, adadelta, adamax, nadam, adagrad
from keras.models import Model
from keras.layers import Input, Dense, Activation

name = ""

#
# def categorical_squared_hinge(y_true, y_pred):
#     """
#     hinge with 0.5*W^2 ,SVM
#     """
#     y_true = 2. * y_true - 1 # trans [0,1] to [-1,1]，注意这个，svm类别标签是-1和1
#     vvvv = K.maximum(1. - y_true * y_pred, 0.) # hinge loss，参考keras自带的hinge loss
# #    vvv = K.square(vvvv) # 文章《Deep Learning using Linear Support Vector Machines》有进行平方
#     vv = K.sum(vvvv, 1, keepdims=False)  #axis=len(y_true.get_shape()) - 1
#     v = K.mean(vv, axis=-1)
#     return v

def compile_model (model : Model):
    # loss=['categorical_squared_hinge']
    model.compile( optimizer = adam,
                   metrics=['accuracy'],
                    loss='sparse_categorical_crossentropy',)

def main_process():
    a = Input(shape=(32,))
    model = Sequential([
        Dense(32, name="A-1", input_shape=(784,)),
        Activation('relu'),
        Dense(10, name="A-2",),
        Activation('softmax'),
    ])
    compile_model(model)
    x = [1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2]
    y = model.predict(x)
    print(y)


if __name__ == "__main__":
    main_process()