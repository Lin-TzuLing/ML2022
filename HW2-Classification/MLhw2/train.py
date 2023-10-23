import numpy as np
from augment_dataset import get_dataset, label_dict
from datetime import datetime
from keras.utils import np_utils
import keras
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt


model_output_path = "model.h5"

print("read data start =", datetime.now().strftime("%H:%M:%S"))
train_data, train_label = get_dataset(save=False, load=True, load_flag='Train')
valid_data, valid_label = get_dataset(save=False, load=True, load_flag='Valid')
print("read data done =", datetime.now().strftime("%H:%M:%S"))

num_classes = len(label_dict)
train_label = keras.utils.np_utils.to_categorical(train_label, num_classes)
valid_label = keras.utils.np_utils.to_categorical(valid_label, num_classes)


def myModel(input_shape):
    model = Sequential()
    # conv 1
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same',
                     activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    # conv 2
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    # conv 3
    model.add(Conv2D(filters=86, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=86, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # linear projection
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    return model

# build model
model = myModel(np.shape(train_data)[1:])
model.summary()

lr = 1e-3
def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))
sgd = SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(learning_rate=lr)
optim = RMSprop(lr=0.001, decay=1e-6)
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=['accuracy'])

callback_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
# 設定模型儲存條件
callback_ckpt = ModelCheckpoint(model_output_path, verbose=1,
                          monitor='val_loss', save_best_only=True, mode='min')
# callback_lrScheduler = LearningRateScheduler(lr_schedule)
callback_lrScheduler = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1,
                                            factor=0.5, min_lr=0.00001)


history = model.fit(train_data, train_label,
                    batch_size=32,
                    epochs=50,
                    validation_data=(valid_data, valid_label),
                    shuffle=True,
                    callbacks=[callback_stop, callback_ckpt, callback_lrScheduler]
                    )

# plot history
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
