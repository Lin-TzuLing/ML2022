from datetime import datetime
from augment_dataset import get_dataset, label_dict
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import confusion_matrix
import seaborn as sns

model_path = 'model.h5'
result_path = 'result.csv'
filter_path = './filter_pic/filter'
matrix_path = 'confusion_matrix.png'

# load best model save in train.py
model = load_model(model_path)
model.summary()

# get_weight
def get_weight_pic(firstCNN_weight, img_size):
    for i in range(firstCNN_weight.shape[-1]):
        tmp_data = firstCNN_weight[...,i]
        output_pic = np.zeros((img_size,img_size,3))
        # interpolation broadcast to resize
        for j in range(tmp_data.shape[-1]):
            channel = scipy.ndimage.zoom(tmp_data[...,j],zoom=20)
            output_pic[...,j] = channel
        # rescale to 0~1
        output_pic = (output_pic - np.min(output_pic)) / np.ptp(output_pic)
        # save image
        plt.imsave(filter_path+'_'+str(i)+'.jpg',
                    output_pic,dpi=20000)
    print('done filter plot, save at {}'.format(filter_path))

# get prediction
def get_predict(test_data,output_path,flag='test'):
    # batch predict test
    y_pred = model.predict(test_data, batch_size=32)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred_class = np.vectorize(label_dict.get)(y_pred)
    if output_path=='None' and flag=='valid':
        return y_pred_class
    elif flag=='test':
        df = pd.DataFrame(data=np.arange(1, len(y_pred_class)+1), columns=['id'])
        df['character'] = y_pred_class
        df.to_csv(output_path,index=False)
        print('done prediction, save at '+output_path)

# get_confusion matrix
def get_confusion(valid_data, valid_label):
    y_pred_valid = get_predict(valid_data, output_path='None', flag='valid')
    valid_label = np.vectorize(label_dict.get)(valid_label)
    cf_matrix = confusion_matrix(valid_label, y_pred_valid, labels=list(label_dict.values()))
    top_margin = 0.04
    bottom_margin = 0.04
    fig, ax = plt.subplots(
        figsize=(30, 30),
        gridspec_kw=dict(top=1 - top_margin, bottom=bottom_margin))

    sns.heatmap(cf_matrix, annot=True,
                fmt='g', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(list(label_dict.values()), rotation = 270)
    ax.yaxis.set_ticklabels(list(label_dict.values()), rotation = 0)
    plt.savefig(matrix_path)
    print('done confusion matrix, save at {}'.format(matrix_path))


if __name__ == "__main__":
    print("load data =", datetime.now().strftime("%H:%M:%S"))
    valid_data, valid_label = get_dataset(save=False, load=True, load_flag='Valid')
    test_data = get_dataset(save=False, load=True, load_flag='Test')
    print("load data done =", datetime.now().strftime("%H:%M:%S"))

    get_confusion(valid_data, valid_label)
    get_weight_pic(firstCNN_weight=model.layers[0].get_weights()[0], img_size=60)
    get_predict(test_data=test_data, output_path=result_path, flag='test')
    print()

