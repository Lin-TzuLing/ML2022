import os
import cv2
import h5py
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


# 讀取train image檔案，train_data_path設定為train data的位置
train_data_path = "./data/theSimpsons-train/train"
test_data_path = "./data/theSimpsons-test/test"
# 儲存dataset的資料夾路徑
output_path = "./dataset/"
img_width, img_height = 48, 48

def contruct_label_dict():
    tmp_dict = {}
    count = 0
    for folder in os.listdir(train_data_path):
        # 讀取不同角色的資料夾，把角色改為用id編號
        tmp_dict[count] = folder
        count += 1
    return tmp_dict

# 建構label_dict，key=id，value=角色名稱
label_dict = contruct_label_dict()
# 要辨識的角色種類
num_class = len(label_dict)

# 把train的圖片讀進來
def get_train_pictures():
    data, label = [], []
    # items in label dict, key=label id, value=original categories
    for key, value in label_dict.items():
        images = [x for x in glob.glob(train_data_path+'/'+value+'/*')]
        print(value + ' train nums :' + str(len(images)))
        for _, image in enumerate(images):
            # read image
            tmp_img = cv2.imread(image)
            # change color channel，在opencv中，默認的顏色排列是BGR，要轉成RGB
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
            # resize back to original height and width
            tmp_img = cv2.resize(tmp_img, (img_height, img_width))
            data.append(tmp_img)
            label.append(key)
    data = np.array(data)
    label = np.array(label)
    return data, label

# 讀test時會打亂順序，重新按照id順序排列test_data
def sort_testData(data,index_arr):
    tmp_data = []
    for count in range(1,len(index_arr)+1):
        idx = np.where(index_arr==count)[0]
        tmp_data.append(data[idx])
    return tmp_data

# 把test的圖片讀進來
def get_test_pictures():
    data, pic_id = [], []
    images = [x for x in glob.glob(test_data_path+'/*')]
    print('test nums :' + str(len(images)))
    for _, image in enumerate(images):
        tmp_img = cv2.imread(image)
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
        tmp_img = cv2.resize(tmp_img, (img_height, img_width))
        data.append(tmp_img)
        pic_id.append(int(image[29:][:-4]))
    data = sort_testData(np.array(data),np.array(pic_id))
    data = np.squeeze(np.array(data))
    return data

# 儲存/讀取資料集(.h5 file)，save:讀原始圖片+存成.h5，load：讀已經存好的.h5
def get_dataset(save=False, load=False, load_flag='Train'):
    if load==True and save==False:
        # augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False
        )

        if load_flag=='Train':
            h5f = h5py.File(output_path+"data.h5", "r")
            train_data = h5f['train_data'][:]
            h5f.close()

            h5f = h5py.File(output_path+"label.h5", "r")
            train_label = h5f['train_label'][:]
            h5f.close()

            # 轉成float32要先做scaling，讓函數以0~1代表0~255
            train_data = train_data.astype("float32") / 255.0
            datagen.flow(train_data)
            print("Train_data : {} , Train_label : {} ".format(train_data.shape, train_label.shape))
            return train_data, train_label

        elif load_flag=='Valid':
            h5f = h5py.File(output_path + "data.h5", "r")
            valid_data = h5f['valid_data'][:]
            h5f.close()

            h5f = h5py.File(output_path + "label.h5", "r")
            valid_label = h5f['valid_label'][:]
            h5f.close()

            # 轉成float32要先做scaling，讓函數以0~1代表0~255
            valid_data = valid_data.astype("float32") / 255.0
            datagen.flow(valid_data)
            print("Valid_data : {} , Valid_label : {} ".format(valid_data.shape, valid_label.shape))
            return valid_data, valid_label

        elif load_flag=='Test':
            # test沒有label，只需要load test_data
            h5f = h5py.File(output_path + "data.h5", "r")
            test_data = h5f['test_data'][:]
            h5f.close()

            # 轉成float32要先做scaling，讓函數以0~1代表0~255
            test_data = test_data.astype("float32") / 255.0
            print("Test_data : {} ".format(test_data.shape))
            return test_data


    elif save==True and load==False:
        # 取得train圖片和label
        train_data, train_label = get_train_pictures()
        # train, valid split
        train_data, valid_data, train_label, valid_label = train_test_split(train_data, train_label,
                                                                            shuffle=True, stratify=train_label,
                                                                            random_state=1111, test_size=0.15)
        # 取得test圖片
        test_data = get_test_pictures()

        h5f = h5py.File(output_path+"data.h5", "w")
        h5f.create_dataset("train_data", data=train_data)
        h5f.create_dataset("valid_data", data=valid_data)
        h5f.create_dataset("test_data", data=test_data)
        h5f.close()

        h5f = h5py.File(output_path+"label.h5", "w")
        h5f.create_dataset("train_label", data=train_label)
        h5f.create_dataset("valid_label", data=valid_label)
        h5f.close()

        print('Successfully save dataset, path = '+output_path)

    else:
        print('error in get_dataset')

if __name__ == "__main__":
    get_dataset(save=True, load=False)