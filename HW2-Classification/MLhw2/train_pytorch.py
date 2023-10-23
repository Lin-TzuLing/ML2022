import numpy as np
from augment_dataset import get_dataset
from dataset import label_dict
from datetime import datetime
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
# from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16


model_output_path = 'xception_model.h5'
print("h5 start =", datetime.now().strftime("%H:%M:%S"))
train_data, train_label = get_dataset(save=False, load=True, load_flag='Train')
print("h5 end =", datetime.now().strftime("%H:%M:%S"))


# train, valid split
train_data, valid_data, train_label, valid_label = train_test_split(train_data, train_label,
                                                    shuffle=True, stratify=train_label,
                                                    random_state=1111,  test_size=0.15)
num_classes = len(label_dict)
train_label = keras.utils.np_utils.to_categorical(train_label, num_classes)
valid_label = keras.utils.np_utils.to_categorical(valid_label, num_classes)

# pretrain model setting
# pretrain_model = Xception(include_top=False, weights='imagenet')



def myModel(input_shape):
    pretrain_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    # freeze all layers
    for layer in pretrain_model.layers:
        layer.trainable = False
    # train last N layers
    for layer in pretrain_model.layers[9:]:
    # for layer in pretrain_model.layers[-8:]:
        layer.trainable = True
    pretrain_model.summary()
    model = Sequential()
    # resize
    # model.add(keras.layers.Resizing(64, 64, interpolation="bilinear",
    #                                 crop_to_aspect_ratio=False, input_shape=input_shape))
    # add pretrain model
    model.add(pretrain_model)
    # layer
    # model.add(GlobalAveragePooling2D())

    # linear projection
    model.add(Flatten())
    # model.add(Dense(1024, activation = "relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    return model


model = myModel(np.shape(train_data)[1:])
# model = myModel((48,48,3))
model.summary()

lr = 1e-3

def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))
sgd = SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(learning_rate=lr)
optim = RMSprop(learning_rate=0.0005, decay=1e-6)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['accuracy'])

callback_stop = EarlyStopping(monitor='val_loss', patience=6, mode='min', verbose=1)
# 設定模型儲存條件
callback_ckpt = ModelCheckpoint(model_output_path, verbose=1,
                          monitor='val_loss', save_best_only=True, mode='min')
# callback_lrScheduler = LearningRateScheduler(lr_schedule)
# callback_lrScheduler = ReduceLROnPlateau(monitor='val_accuracy',
callback_lrScheduler = ReduceLROnPlateau(monitor='val_loss',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)



history = model.fit(
# history = model.fit_generator(datagen.flow(train_data,train_label, batch_size=32, shuffle=True),
                     train_data, train_label,
                    batch_size=64,
                    epochs=150,
                    validation_data=(valid_data, valid_label),
                    # steps_per_epoch=len(train_data) // 32,
                    shuffle=True,
                    # callbacks=[callback_stop, callback_ckpt]
                    callbacks=[callback_stop, callback_ckpt, callback_lrScheduler]
                    )

def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics), '-o')
    plt.plot(history.history.get(val_metrics), '-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plot_train_history(history, 'loss', 'val_loss')
plt.subplot(1, 2, 2)
plot_train_history(history, 'accuracy', 'val_accuracy')
plt.show()

del train_data, train_label, model, model_output_path