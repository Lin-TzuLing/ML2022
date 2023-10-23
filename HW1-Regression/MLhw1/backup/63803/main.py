import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('processed_data_path',type=str, help='設定data來源路徑')
parser.add_argument('log_name', type=str, help='設定log name')
parser.add_argument('log_path', type=str, help='設定loge儲存路徑')
parser.add_argument('result_path', type=str, help='設定result儲存路徑')
args = parser.parse_args()
print(args)

# read data
PROCESSED_DATA_PATH = args.processed_data_path
train = pd.read_csv(PROCESSED_DATA_PATH+'train.csv')
valid = pd.read_csv(PROCESSED_DATA_PATH+'valid.csv')
test = pd.read_csv(PROCESSED_DATA_PATH+'test.csv')
# dataframe to tensor
def prepare_tensor(df, labeled=True):
    if labeled==True:
        data = torch.tensor(df.drop(['id','price'],axis=1).values)
        label = torch.tensor(df['price'].values).unsqueeze(dim=1)
        return data, label
    elif labeled==False:
        data = torch.tensor(df.drop(['id'],axis=1).values)
        return data
    else:
        print('error')

train_data, train_label = prepare_tensor(train, labeled=True)
valid_data, valid_label = prepare_tensor(valid, labeled=True)
test_data = prepare_tensor(test, labeled=False)
print('data prepared')


# build model
class myModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(myModel, self).__init__()
        linear_layer = []
        activate_layer = []
        deep_layer = []
        deeper_layer = []

        # block 1
        self.l1 = nn.Linear(input_dim, output_dim)

        # block 2
        self.l3 = nn.Linear(input_dim, 256)
        self.dropout2 = nn.Dropout(0.1)
        self.relu1 = nn.ReLU()
        self.l4 = nn.Linear(256,256)
        self.l5 = nn.Linear(256,128)
        self.l6 = nn.Linear(128,1)

        # block 3
        self.l7 = nn.Linear(input_dim, 1024)
        self.dropout3 = nn.Dropout(0.3)
        self.relu2 = nn.ReLU()
        self.l8 = nn.Linear(1024,1024)
        self.l9 = nn.Linear(1024,512)
        self.l10 = nn.Linear(512, 256)
        self.l11 = nn.Linear(256, 128)
        self.l12 = nn.Linear(128, 32)
        self.l13 = nn.Linear(32, 16)
        self.l14 = nn.Linear(16, 1)

        self.l15 = nn.Linear(input_dim, 128)
        self.l16 = nn.Linear(128,64)
        self.l17 = nn.Linear(64,1)

        self.l18 = nn.Linear(64,32)
        self.l19 = nn.Linear(32, 32)
        # self.l19 = nn.Linear(32, 1)
        self.l20 = nn.Linear(32,1)
        self.l21 = nn.Linear(128,128)


        linear_layer += [self.l1]
        # activate_layer += [self.l15, self.relu2, self.l16,self.relu2,
        #                    self.l18, self.relu2, self.l19, self.l20]
        activate_layer += [self.l15, self.dropout3, self.relu2,
                           self.l21, self.dropout3, self.relu2,
                           self.l16, self.dropout3,self.relu2,
                          self.l18, self.dropout3, self.relu2,
                           self.l19, self.l20]
        deep_layer += [self.l7, self.dropout3,
                       self.l8, self.dropout3,
                       self.l9, self.dropout3,
                       self.l10, self.dropout3,
                       self.l11, self.dropout3,
                       self.l17, self.l18, self.l19]
        deeper_layer += []

        self.linear = nn.Sequential(*linear_layer)
        self.activate = nn.Sequential(*activate_layer)
        self.deep = nn.Sequential(*deep_layer)
        self.deeper = nn.Sequential(*deeper_layer)


    def forward(self, x, flag='deep'):
        x = x.float()
        if flag=='linear':
            out = self.linear(x)
            return out
        elif flag=='activate':
            out = self.activate(x)
            return out
        elif flag=='deep':
            out = self.deep(x)
            return out
        elif flag=='deeper':
            out = self.deeper(x)
            return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

n_feature = len(list(train.columns))-len(['id','price'])
model = myModel(input_dim=n_feature, output_dim=1)
# initialize weight, bias of layer in model
model.initialize_weights()


def l1_regularizer(model, lambda_l1=0.01):
    lossl1 = 0
    for model_param_name, model_param_value in model.named_parameters():
            if model_param_name.endswith('weight'):
                lossl1 += lambda_l1 * model_param_value.abs().sum()
    return lossl1

def testMAE(flag):
    model.eval()
    criterion = nn.L1Loss()
    with torch.no_grad():
        valid_pred = model(valid_data, flag=flag)
        valid_pred = torch.tensor(np.expm1(valid_pred.numpy()))
        label = torch.tensor(np.expm1(valid_label.numpy()))
        mae = criterion(valid_pred, label)
    return mae

def predTest(flag):
    model.eval()
    with torch.no_grad():
        pred_test = []
        x = DataLoader(test_data, batch_size=200)
        for _, data in enumerate(x):
            # 預測unlabeled test data的price
            pred_test.append(model(data, flag=flag))
        pred_test = torch.cat(pred_test)
    return pred_test

def trainModel(epochs, batch_size , lr):
    # model type
    model_type = 'activate'

    # loss and optimizer
    learning_rate = lr
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_MAE = 100000000
    LOG_NAME = args.log_name

    # batch
    batch_data = DataLoader(train_data, batch_size=batch_size)
    batch_label = DataLoader(train_label, batch_size=batch_size)

    # training loop
    print('start training')

    for epoch in tqdm(range(epochs)):

        model.train()
        running_loss = 0

        # batch training
        for i, (data, label) in enumerate(zip(batch_data, batch_label)):
            pred_train = model(data, flag=model_type)

            optimizer.zero_grad()
            loss = criterion(pred_train, label.float())
            # loss = float(loss) + l1_regularizer(model, 0.08)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # train 1 epoch test 1 次
        mae = testMAE(model_type)
        writer.add_scalar((LOG_NAME+'/MAE'+'_'+str(model_type)+'_'+str(batch_size)), mae, epoch+1)
        writer.add_scalar((LOG_NAME+'/Loss'+'_'+str(model_type)+'_'+str(batch_size)), running_loss/i, epoch+1)
        if mae < best_MAE:
            pred_test = predTest(model_type)
            best_MAE = mae
            tqdm.write('epoch {}, loss {}, mae {}'.format(epoch + 1, running_loss / i, mae))
    # 回傳最好的那個pred test
    return pred_test


if __name__ == '__main__':
    LOG_PATH = args.log_path
    writer = SummaryWriter(LOG_PATH)
    pred_test = trainModel(epochs=330, batch_size=128, lr=0.001)
    print('Finished Training')
    ans_df = pd.DataFrame(data=np.arange(1, len(pred_test)+1), columns=['id'])
    ans_df['price'] = pred_test.numpy()
    flag_log = True
    if flag_log==True:
        ans_df['price'] = ans_df['price'].apply(lambda x: np.expm1(x))
    RESULT_PATH = args.result_path
    ans_df.to_csv(RESULT_PATH, index=False)
    print('done')


# tensorboard --logdir=./log --port 8123
