import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from jiwer import wer
from keras.models import load_model
import argparse
import os
os.environ["CUDA_VISIABLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


parser = argparse.ArgumentParser()
parser.add_argument("train_path",type=str, help="training data path")
parser.add_argument("label_path",type=str, help="training data label path")
parser.add_argument("model_path",type=str, help="store model based on validation CTC loss")
parser.add_argument("wer_model_path",type=str, help="store model based on wer score")
args = parser.parse_args()


# path
train_path = args.train_path
label_path = args.label_path
model_path = args.model_path
wer_model_path = args.wer_model_path

label_df = pd.read_csv(label_path)
label_df = label_df.applymap(str)


# The set of characters accepted in the transcription.
characters = [x for x in 'abcdefghijklmnopqrstuvwxyz ']
# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)
if __name__ == "__main__":
    print("The vocabulary is: {}".format(char_to_num.get_vocabulary()))
    print("size : {}".format(char_to_num.vocabulary_size()))
    print("Size of the whole set: {}".format(len(label_df)))

# delete data with label noise (not pure alphabets)
delete_id = set()
for i in range(len(label_df)):
    for character in label_df.iloc[i]['text']:
        if character.lower() not in characters:
            delete_id.add(i)

label_df = label_df.drop(label_df.index[list(delete_id)])
split = int(len(label_df) * 0.90)
label_train = label_df[:split]
label_valid = label_df[split:]

if __name__ == "__main__":
    print("Size of the modified set: {}".format(len(label_df)))
    print("Size of the train set: {}".format(len(label_train)))
    print("Size of the valid set: {}".format(len(label_valid)))

# An integer scalar Tensor. The window length in samples.
frame_length = 128
# An integer scalar Tensor. The number of samples to step.
frame_step = 128
# An integer scalar Tensor. The size of the FFT to apply.
fft_length = 128

# audio preprocess
def encode_single_sample(filename, label):
    # --------------------------process audio ----------------------------
    # 1. Read wav file
    file = tf.io.read_file(train_path + filename + ".wav")
    # 2. Decode the wav file
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    # 5. Only need the magnitude, applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 6. normalization
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    # --------------------------process label ----------------------------
    # 7. Convert label to Lower case
    label = tf.strings.lower(label)
    # 8. Split the label
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    # 9. Map the characters in label to numbers
    label = char_to_num(label)
    # 10. Return a dict as our model is expecting two inputs
    return spectrogram, label


batch_size = 8
lr = 1e-3

# training dataset
train_dataset = tf.data.Dataset.from_tensor_slices(
    (list(label_train["id"]), list(label_train["text"]))
)
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Define the validation dataset
validation_dataset = tf.data.Dataset.from_tensor_slices(
    (list(label_valid["id"]), list(label_valid["text"]))
)
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)


# CTC loss function
def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# model construction
def myModel(input_dim, output_dim, rnn_layers=3, rnn_units=128):
    # input
    input_spectrogram = layers.Input((None, input_dim), name="input")
    # Expand the dimension to use 2D CNN.
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    # Conv 1
    x = layers.Conv2D(filters=32,kernel_size=[11, 41], strides=[2, 2],
                      padding="same", use_bias=False, name="conv_1")(x)
    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)
    # Conv 2
    x = layers.Conv2D(filters=32, kernel_size=[11, 21], strides=[1, 2],
                      padding="same", use_bias=False, name="conv_2")(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)
    # Reshape
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    # Dense layer
    x = layers.Dense(units=1024, name="dense_1")(x)
    x = layers.ReLU(name="dense_1_relu")(x)
    x = layers.Dropout(rate=0.5)(x)
    # RNN layer
    for i in range(1, rnn_layers + 1):
        LSTM = layers.LSTM(units=rnn_units, activation="tanh", recurrent_activation="sigmoid",
                               use_bias=True, name=f"lstm_{i}",return_sequences=True)
        x = layers.Bidirectional(LSTM, name=f"bidirectional_{i}", merge_mode="concat")(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=0.5)(x)
     # Dense layer
    x = layers.Dense(units=rnn_units * 2, name="dense_2")(x)
    x = layers.ReLU(name="dense_2_relu")(x)
    x = layers.Dropout(rate=0.5)(x)
    # Classification layer
    output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
    # Model
    model = Model(input_spectrogram, output, name="CNN_LSTM_model")
    # Optimizer
    optim = Adam(learning_rate=lr)
    # Compile the model and return
    model.compile(optimizer=optim, loss=CTCLoss)
    return model


# build model
model = myModel(input_dim=fft_length // 2 + 1,
                output_dim=char_to_num.vocabulary_size(),
                rnn_units=512)

model.summary(line_length=110)

# decode output of model
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text

# callback for validation wer score and save wer model
wer_history = [1.0]
class CallbackEval(keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                targets.append(label)
        wer_score = wer(targets, predictions)
        if wer_score < min(wer_history):
            model.save(wer_model_path)
            print('save wer model')
        wer_history.append(wer_score)
        print("-" * 100)
        print("Word Error Rate: {}".format(wer_score))
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print("Target : {}".format(targets[i]))
            print("Prediction: {}".format(predictions[i]))
            print("-" * 100)

# plot history
def plot_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics), '-o')
    plt.plot(history.history.get(val_metrics), '-r')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train','validation'])


callback_stop = EarlyStopping(monitor='val_loss', patience=25, mode='min', verbose=1)
callback_ckpt = ModelCheckpoint(model_path, verbose=1,
                                monitor='val_loss', save_best_only=True, mode='min')
callback_lrScheduler = ReduceLROnPlateau(monitor='val_loss', patience=5,
                                         verbose=1, factor=0.5, min_lr=0.00001)
# Define the number of epochs.
epochs = 100
# Callback function to check transcription on the val set.
validation_callback = CallbackEval(validation_dataset)

load_flag = False
if __name__ == "__main__":
    if load_flag == True:
        model = load_model(model_path, custom_objects={'CTCLoss': CTCLoss})

    # Train model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[validation_callback, callback_stop, callback_ckpt, callback_lrScheduler],
    )


    # plot history
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plot_history(history, 'loss', 'val_loss')
    plt.subplot(1,2,2)
    wer_history = wer_history[1:]
    plt.plot(list(np.arange(len(wer_history))),wer_history)
    plt.ylabel('wer score')
    plt.xlabel('Epochs')
    plt.legend(['valid'])
    plt.savefig("history.png")

    # check results on validation samples
    predictions = []
    targets = []
    for batch in validation_dataset:
        X, y = batch
        batch_predictions = model.predict(X)
        batch_predictions = decode_batch_predictions(batch_predictions)
        predictions.extend(batch_predictions)
        for label in y:
            label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
            targets.append(label)
    wer_score = wer(targets, predictions)
    print("-" * 100)
    print(f"Word Error Rate: {wer_score:.4f}")
    print("-" * 100)
    for i in np.random.randint(0, len(predictions), 5):
        print(f"Target    : {targets[i]}")
        print(f"Prediction: {predictions[i]}")
        print("-" * 100)

    print('training done')