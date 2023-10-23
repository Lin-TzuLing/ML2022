import pandas as pd
import tensorflow as tf
from train_old import frame_length, frame_step, fft_length, batch_size
from train_old import model_path, wer_model_path, decode_batch_predictions, CTCLoss
import numpy as np
from keras.models import load_model
import argparse
import os
os.environ["CUDA_VISIABLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument("test_path",type=str, help="testing data path")
parser.add_argument("output_path",type=str, help="store model based on validation CTC loss")
parser.add_argument("output_wer_path",type=str, help="store model based on wer score")
args = parser.parse_args()

# path
test_path = args.test_path
output_path = args.output_path
output_wer_path = args.output_wer_path

# audio preprocess
def encode_single_sample(filename):
    file = tf.io.read_file(test_path + filename + ".wav")
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    return spectrogram


id_list = []
for filename in os.listdir(test_path):
    if filename.endswith(".wav"):
        id_list.append(filename[:-4])
    else:
        print(filename)

id_list = sorted([int(x) for x in id_list])
id_list = [str(x) for x in id_list]

# test dataset
test_dataset = tf.data.Dataset.from_tensor_slices(id_list)
test_dataset = (
    test_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

def predict(load_model_path):
    model = load_model(load_model_path, custom_objects={'CTCLoss':  CTCLoss})
    predictions = []
    for batch in test_dataset:
        X = batch
        batch_predictions = model.predict(X)
        batch_predictions = decode_batch_predictions(batch_predictions)
        predictions.extend(batch_predictions)
    for i in np.random.randint(0, len(predictions), 2):
        print("Prediction: {}".format(predictions[i]))
        print("-" * 100)
    return predictions

if __name__ == "__main__":
    m = predict(model_path)
    wer_m = predict(wer_model_path)
    dict_m = {'id': id_list, 'text': m}
    df_m = pd.DataFrame(dict_m)
    dict_wer = {'id': id_list, 'text': wer_m}
    df_wer = pd.DataFrame(dict_wer)

    df_m.to_csv(output_path, index=False)
    df_wer.to_csv(output_wer_path, index=False)
print()