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
        data = torch.tensor(df.drop(['id','price','zipcode'],axis=1).values)
        zipcode = torch.tensor(df['zipcode'].values)
        label = torch.tensor(df['price'].values).unsqueeze(dim=1)
        return data, zipcode, label
    elif labeled==False:
        data = torch.tensor(df.drop(['id','zipcode'],axis=1).values)
        zipcode = torch.tensor(df['zipcode'].values)
        return data, zipcode
    else:
        print('error')

train_data, train_zipcode, train_label = prepare_tensor(train, labeled=True)
valid_data, valid_zipcode, valid_label = prepare_tensor(valid, labeled=True)
test_data, test_zipcode = prepare_tensor(test, labeled=False)
print('data prepared')



# build model
class myModel(nn.Module):
    def __init__(self, input_dim, output_dim, zipcode_dim=5):
        super(myModel, self).__init__()
        linear_layer = []
        linear1_layer = []

        self.l1 = nn.Linear((input_dim+zipcode_dim), 128)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(64, 32)
        self.l5 = nn.Linear(32, 32)
        self.l6 = nn.Linear(32, output_dim)
        self.l7 = nn.Linear(128,128)
        self.l8 = nn.Linear(128,64)
        self.dropout1 = nn.Dropout(0.3)
        self.relu = nn.ReLU()


        # zipcode embedding
        self.zip_embed = nn.Embedding(200,zipcode_dim)
        linear1_layer += [self.l1, self.dropout1, self.relu,
                          self.l7, self.dropout1, self.relu,
                          self.l8, self.dropout1, self.relu,
                          self.l4, self.dropout1, self.relu,
                          self.l5, self.l6]
        # linear_layer += [self.l1, self.dropout1,  self.relu,
        #                  self.l2, self.dropout1, self.relu,
        #                  self.l3, self.dropout1,self.relu,
        #                  self.l4, self.dropout1, self.relu,
        #                  self.l5, self.l6,]
        self.linear = nn.Sequential(*linear_layer)
        self.linear1 = nn.Sequential(*linear1_layer)

    def forward(self, x, zipcode):
        x = x.float()
        zipcode = zipcode.long()
        zipcode = self.zip_embed(zipcode)
        out = torch.cat((x,zipcode), 1)
        # out = self.linear(out)
        out = self.linear1(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                # nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()

n_feature = len(list(train.columns))-len(['id','price','zipcode'])
model = myModel(input_dim=n_feature, output_dim=1, zipcode_dim=5)
# initialize weight, bias of layer in model
model.initialize_weights()


def l1_regularizer(model, lambda_l1=0.01):
    lossl1 = 0
    for model_param_name, model_param_value in model.named_parameters():
            if model_param_name.endswith('weight'):
                lossl1 += lambda_l1 * model_param_value.abs().sum()
    return lossl1

def testMAE():
    model.eval()
    criterion = nn.L1Loss()
    with torch.no_grad():
        valid_pred = model(valid_data, valid_zipcode)
        valid_pred = torch.tensor(np.expm1(valid_pred.numpy()))
        label = torch.tensor(np.expm1(valid_label.numpy()))
        mae = criterion(valid_pred, label)
    return mae

def predTest():
    model.eval()
    with torch.no_grad():
        pred_test = []
        x = DataLoader(test_data, batch_size=200)
        x_zip = DataLoader(test_zipcode, batch_size=200)
        for _, (data, zipcode) in enumerate(zip(x, x_zip)):
            # 預測unlabeled test data的price
            pred_test.append(model(data, zipcode))
        pred_test = torch.cat(pred_test)
    return pred_test

def trainModel(epochs, batch_size , lr):
    # loss and optimizer
    learning_rate = lr
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_MAE = 100000000
    LOG_NAME = args.log_name

    # batch
    batch_data = DataLoader(train_data, batch_size=batch_size)
    batch_zipcode = DataLoader(train_zipcode, batch_size=batch_size)
    batch_label = DataLoader(train_label, batch_size=batch_size)

    # training loop
    print('start training')

    for epoch in tqdm(range(epochs)):

        model.train()
        running_loss = 0

        # batch training
        for i, (data, zipcode, label) in enumerate(zip(batch_data, batch_zipcode, batch_label)):
            pred_train = model(data,zipcode)
            optimizer.zero_grad()
            loss = criterion(pred_train, label.float())
            # loss = float(loss) + l1_regularizer(model, 0.08)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # train 1 epoch test 1 次
        mae = testMAE()
        writer.add_scalar((LOG_NAME+'/MAE'+'_'+str(batch_size)), mae, epoch+1)
        writer.add_scalar((LOG_NAME+'/Loss'+'_'+str(batch_size)), running_loss/i, epoch+1)
        if mae < best_MAE:
            pred_test = predTest()
            best_MAE = mae
            # torch.save(model.state_dict(), "./best_model.pt")
            tqdm.write('epoch {}, loss {}, mae {}'.format(epoch + 1, running_loss / i, mae))
    # 回傳最好的那個pred test
    return pred_test


if __name__ == '__main__':
    LOG_PATH = args.log_path
    writer = SummaryWriter(LOG_PATH)
    pred_test = trainModel(epochs=350, batch_size=128, lr=0.001)
    print('Finished Training')
    ans_df = pd.DataFrame(data=np.arange(1, len(pred_test)+1), columns=['id'])
    ans_df['price'] = pred_test.numpy()
    flag_log = True
    if flag_log==True:
        ans_df['price'] = ans_df['price'].apply(lambda x: np.expm1(x))
    RESULT_PATH = args.result_path
    ans_df.to_csv(RESULT_PATH,index=False)
    print('done')


# tensorboard --logdir=./log --port 8123
