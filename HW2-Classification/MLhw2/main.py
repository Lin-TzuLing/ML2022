import numpy as np
from dataset import get_dataset
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# from torch.metrics import Accuracy

model_output_path = './model/best_model.pth'

print("h5 start =", datetime.now().strftime("%H:%M:%S"))
train_data, train_label, test_data = get_dataset(save=False, load=True)
print("h5 end =", datetime.now().strftime("%H:%M:%S"))
train_data, train_label, test_data = torch.from_numpy(train_data), torch.from_numpy(train_label), torch.from_numpy(test_data)
# reshape data to (N,C,H,W)
train_data = torch.reshape(train_data, (train_data.size()[0], 3, 42, 42))
test_data = torch.reshape(test_data, (test_data.size()[0], 3, 42, 42))

# train, valid split
train_data, valid_data, train_label, valid_label = train_test_split(train_data, train_label,
                                                    shuffle=True, stratify=train_label,
                                                    random_state=1111,  test_size=0.15)
num_class = 50

class myModel(nn.Module):
    def __init__(self, input_channel, output_dim):
        super(myModel, self).__init__()
        layer = []
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # block 1
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=3, padding='same')
        # self.bn1 = nn.BatchNorm2d(num_features=4, affine=True)
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        # self.pool1 = nn.AdaptiveAvgPool2d(24)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # linear
        self.linear = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=output_dim, bias=True),
            nn.Softmax(dim=1)
        )


        layer += [self.conv1, self.relu,
                  self.conv1_2, self.relu,
                  self.pool1, self.dropout]
        layer += [self.conv2, self.relu,
                  self.conv2_2, self.relu,
                  self.pool2, self.dropout]
        layer += [self.conv3, self.relu,
                  self.conv3_2, self.relu,
                  self.pool3, self.dropout]
        self.conv = nn.Sequential(*layer)

    def forward(self, x):
        out = self.conv(x)
        out = self.linear(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


model = myModel(input_channel=train_data.size()[1], output_dim=num_class)
# model.initialize_weights()


def validAcc(batch_size):
    model.eval()
    with torch.no_grad():
        data_valid = DataLoader(valid_data, batch_size=32)
        label_valid = DataLoader(valid_label, batch_size=32)
        count_true = 0.0
        #  batch valid
        for i, (data, label) in enumerate(zip(data_valid, label_valid)):
            # valid_data通過model得到test data 分類
            pred_valid = model(data)
            pred_valid = torch.argmax(pred_valid, dim=1)
            count_true += pred_valid.eq(label).float()
    tqdm.write(str(count_true))
    return count_true/len(valid_label)


def trainModel(epochs, batch_size , lr):

    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                weight_decay=1e-6, momentum=0.9, nesterov=True)
    lambda1 = lambda epoch: lr * (0.1 ** int(epoch / 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    # LOG_NAME = args.log_name

    # batch
    batch_data = DataLoader(train_data, batch_size=batch_size)
    batch_label = DataLoader(train_label, batch_size=batch_size)

    # training loop
    tqdm.write('start training')
    for epoch in tqdm(range(epochs)):

        model.train()
        running_loss = 0

        # batch training
        for i, (data, label) in enumerate(zip(batch_data, batch_label)):
            optimizer.zero_grad()
            pred_train = model(data)
            # target need to be class indice
            loss = criterion(pred_train, label.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        # train 1 epoch test 1 次
        acc = validAcc(batch_size)
        # writer.add_scalar((LOG_NAME+'/MAE'+'_'+str(batch_size)), mae, epoch+1)
        # writer.add_scalar((LOG_NAME+'/Loss'+'_'+str(batch_size)), running_loss/i, epoch+1)
        tqdm.write('epoch {}, loss {}, acc {}'.format(epoch + 1, running_loss / i, acc))
        if acc > best_acc:
            torch.save(model.state_dict(), model_output_path)
            best_acc = acc
            tqdm.write('save at here')


if __name__ == "__main__":
    trainModel(epochs=10, batch_size=32, lr= 0.01)
    print('Finished Training')
    pred_embed = pred_embed.detach().numpy()
    with open('./0712118.npy', 'wb') as file:
        np.save(file, pred_embed)
