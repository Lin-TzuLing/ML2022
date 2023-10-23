import pandas as pd
import os
os.environ["CUDA_VISIABLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
from utils.Dataset import TrainDataset, TestDataset
from utils.Tokenizer import ch_Tokenizer, tl_Tokenizer
from jiwer import wer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dict_path",type=str, help="dictionary path")
parser.add_argument("train_path",type=str, help="training data path")
parser.add_argument("test_path",type=str, help="testing data path")
parser.add_argument("model_path",type=str, help="store best model.pt")
parser.add_argument("output_path",type=str, help="store testing output")
args = parser.parse_args()

# path and hypers
CH_MAXLENGTH = 50
TL_MAXLENGTH = 50
dict_path = args.dict_path
train_path = args.train_path
test_path = args.test_path
model_dir = args.model_path
output_path = args.output_path


ch_Tokenizer = ch_Tokenizer(dict_path)
tl_Tokenizer = tl_Tokenizer(dict_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoder ch to latent (source language -> latent representation)
class Encoder(nn.Module):
    def __init__(self, encoder_embedding_num, encoder_hidden_num, ch_vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(ch_vocab_size, encoder_embedding_num)
        # 2 layers of stacked RNN (3 layer will cause overfitting)
        self.rnn = nn.GRU(encoder_embedding_num, encoder_hidden_num, num_layers=2, batch_first=True)

    def forward(self, en_index):
        en_embedding = self.embedding(en_index)
        encoder_output, encoder_hidden = self.rnn(en_embedding)
        return encoder_output, encoder_hidden

# Attention mechanism for decoder
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, decoder_state_t, encoder_outputs):
        b, s, h = encoder_outputs.shape
        attention_scores = torch.sum(
         torch.tile(decoder_state_t.unsqueeze(dim=1), dims=(s,1)) * encoder_outputs,dim=-1)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        context = torch.sum(attention_scores.unsqueeze(dim=-1) * encoder_outputs, dim=1)
        return context, attention_scores

# Decoder latent to tl (latent representation -> target language)
class AttentionDecoder(nn.Module):
    def __init__(self,
                 decoder_embedding_num, decoder_hidden_num,
                 tl_vocab_size, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(tl_vocab_size, decoder_embedding_num)
        self.gru = nn.GRUCell(decoder_embedding_num, decoder_hidden_num)
        self.attention = Attention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_input, encoder_hidden, encoder_output):
        embed = self.embedding(decoder_input)
        b, s, h = embed.shape
        ht = encoder_hidden[0]
        decoder_output = []
        for t in range(s):
            decoder_input = embed[:, t, :]
            ht = self.gru(decoder_input, ht)
            context, attention_probs = self.attention(ht, encoder_output)
            ht = self.dropout(ht)
            yt = torch.cat((ht, context), dim=-1)
            decoder_output.append(yt)
        decoder_output = torch.stack(decoder_output, dim=0)
        decoder_output = decoder_output.transpose(0, 1)
        return decoder_output

# Seq2Seq translation model (encoder+decoder)
class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder_embedding_num, encoder_hidden_num, ch_vocab_size,
                 decoder_embedding_num, decoder_hidden_num, tl_vocab_size,
                 device='cuda', dropout=0.3):
        super().__init__()
        self.device = device
        self.encoder = Encoder(encoder_embedding_num, encoder_hidden_num, ch_vocab_size)
        self.decoder = AttentionDecoder(decoder_embedding_num, decoder_hidden_num, tl_vocab_size, dropout)
        self.projection = nn.Linear(2 * decoder_hidden_num, tl_vocab_size)
        self.tl_tokenizer = tl_Tokenizer

    # teacher forcing training (for train session)
    def forward(self, ch_index, tl_index):
        encoder_outputs, encoder_hidden = self.encoder(ch_index)
        decoder_output = self.decoder(tl_index, encoder_hidden, encoder_outputs)
        # NO SOFTMAX for activation (cross-entropy doesn't need softmax first)
        return self.projection(decoder_output)

    # no teacher forcing, take last prediction as new input (for valid and test session)
    def inference(self, ch_index, max_length=TL_MAXLENGTH):
        with torch.no_grad():
            encoder_output, encoder_hidden = self.encoder(ch_index)
            decoder_input = torch.tensor([[self.tl_tokenizer.BOS]]).to(self.device)
            ht = encoder_hidden[0].to(self.device)
            predictions = []
            for t in range(max_length):
                embed = self.decoder.embedding(decoder_input)[:, 0, :]
                ht = self.decoder.gru(embed, ht)
                context, _ = self.decoder.attention(ht, encoder_output)
                yt = torch.cat((ht, context), dim=-1)
                # NO SOFTMAX for activation (cross-entropy doesn't need softmax first)
                pred = self.projection(yt)
                w_index = int(torch.argmax(pred, dim=-1))
                word = self.tl_tokenizer.decode_single(w_index)
                if word == "<EOS>":
                    break
                predictions.append(word)
                decoder_input = torch.tensor([[w_index]]).to(self.device)
            return "".join(predictions)


# Split train and valid data (with fixed batch_size and
# batch_size=1 for valid because not using teacher forcing)
def train_val_split(dataset, batch_size, num_workers, validation_split=0.2):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SequentialSampler(train_indices)
    valid_sampler = torch.utils.data.SequentialSampler(val_indices)
    train_iter = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers,
                            collate_fn=dataset.collate_fn)
    valid_iter = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=1, num_workers=num_workers,
                            collate_fn=dataset.collate_fn)
    return train_iter, valid_iter

# Build model
model = Seq2Seq(encoder_embedding_num=1024, encoder_hidden_num=512, ch_vocab_size=ch_Tokenizer.length(),
                decoder_embedding_num=1024, decoder_hidden_num=512, tl_vocab_size=tl_Tokenizer.length(),
                device='cuda', dropout=0.5)

# Prediction for test data (load best model)
def test(model, load_dir, output_path):
    dataset = TestDataset(test_path, ch_tokenizer=ch_Tokenizer, nums=None)
    idlist = dataset.find_idlist()
    test_iter, _ = train_val_split(dataset, batch_size=1, num_workers=0, validation_split=0.0)

    model = model
    model.load_state_dict(copy.deepcopy(torch.load(load_dir + 'weights/best.pt')))
    model.to(device)
    model.eval()
    prediction = []
    for ch_index in tqdm(test_iter):
        ch_index = ch_index.to(device)
        pred = model.inference(ch_index)
        prediction.append(pred)
    ans_df = pd.DataFrame(list(zip(idlist, prediction)), columns =['id', 'txt'])
    ans_df.to_csv(output_path, index=False)

# Training
history_trainloss = []
history_validwer=[]
def train(model, epochs, save_dir):
    dataset = TrainDataset(train_path, ch_tokenizer=ch_Tokenizer, tl_tokenizer=tl_Tokenizer, nums=None)
    train_iter, valid_iter = train_val_split(dataset, batch_size=64, num_workers=0, validation_split=0.1)

    model = model
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True,
                                                           mode='min',min_lr=1e-5, eps=1e-8)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    cross_loss = nn.CrossEntropyLoss()
    cross_loss.to(device)
    best_wer = 3.0

    for e in range(epochs):
        # training
        model.train()
        for ch_index, tl_index in tqdm(train_iter):
            ch_index = ch_index.to(device)
            tl_index = tl_index.to(device)
            pred = model(ch_index, tl_index[:, :-1])
            label = tl_index[:, 1:]
            loss = cross_loss(pred.reshape(-1, pred.shape[-1]), label.reshape(-1))
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            loss.backward()
            optimizer.step()
        # scheduler.step()

        # validation
        model.eval()
        val_wer, n = .0, 0
        for ch_index, tl_index in tqdm(valid_iter):
            ch_index = ch_index.to(device)
            tl_index = tl_index.to(device)
            pred = model.inference(ch_index)
            label = "".join(tl_Tokenizer.decode(tl_index[0].tolist()[1:-1]))
            val_wer += wer(label, pred)
            if n<=1:
                tqdm.write(('pred: {}'.format(pred)))
                tqdm.write(('label:{}'.format(label)))
            n += 1
        val_wer /= n
        print(f"epoch {e}  train loss {loss.item()}  val avg wer score {val_wer}")
        history_trainloss.append(loss.item())
        history_validwer.append(val_wer)

        # save best model
        if val_wer <= best_wer:
            if not os.path.exists(os.path.join(save_dir, 'weights')):
                os.makedirs(os.path.join(save_dir, 'weights'))
            torch.save(model.state_dict(), os.path.join(save_dir, "weights/best.pt"))
            best_wer = val_wer
        # torch.save(model.state_dict(), os.path.join(save_dir, "weights/last.pt"))

        # step learning rate
        model.train()
        scheduler.step(val_wer)

# Plot history
def plot_history(train_loss, valid_wer):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(list(np.arange(len(train_loss))), train_loss)
    plt.ylabel('Cross Entropy Loss')
    plt.xlabel('Epochs')
    plt.legend(['train'])

    plt.subplot(1, 2, 2)
    plt.plot(list(np.arange(len(valid_wer))), valid_wer)
    plt.ylabel('Word Error Rate (WER)')
    plt.xlabel('Epochs')
    plt.legend(['valid'])
    plt.savefig("history.png")


# main
train(model=model, epochs=300, save_dir=model_dir)
test(model=model, load_dir=model_dir, output_path=output_path)
plot_history(history_trainloss, history_validwer)
print()