import pandas as pd
import torch

class TrainDataset(torch.utils.data.Dataset):
    @staticmethod
    def get_data(dataset_path, nums=None):
        datas = pd.read_pickle(dataset_path, compression='gzip')
        ch_data = list(datas["ch"])
        tl_data = list(datas["tl"])
        if nums is None:
            return ch_data, tl_data
        else:
            return ch_data[:nums], tl_data[:nums]

    def __init__(self, dataset_path, ch_tokenizer, tl_tokenizer, nums=None):
        ch_data, tl_data = self.get_data(dataset_path, nums=nums)
        self.ch_data = ch_data
        self.tl_data = tl_data
        self.ch_tokenizer = ch_tokenizer
        self.tl_tokenizer = tl_tokenizer

    def __getitem__(self, index):
        ch = self.ch_data[index]
        tl = self.tl_data[index]
        ch_index = self.ch_tokenizer.encode(sentence=ch)
        tl_index = self.tl_tokenizer.encode(sentence=tl)
        return ch_index, tl_index

    def __len__(self):
        assert len(self.tl_data) == len(self.ch_data)
        return len(self.ch_data)

    def collate_fn(self, batch_list):
        ch_index, tl_index = [], []
        chPAD = self.ch_tokenizer.PAD
        tlPAD = self.tl_tokenizer.PAD
        chBOS = self.ch_tokenizer.BOS
        chEOS = self.ch_tokenizer.EOS
        tlBOS = self.tl_tokenizer.BOS
        tlEOS = self.tl_tokenizer.EOS
        for ch, tl in batch_list:
            # TODO
            # ch_index.append(torch.tensor(ch))
            ch_index.append(torch.tensor([chBOS] + ch + [chEOS]))
            tl_index.append(torch.tensor([tlBOS] + tl + [tlEOS]))
        # from torch.nn.utils.rnn import pad_sequence
        ch_index = torch.nn.utils.rnn.pad_sequence(ch_index, batch_first=True, padding_value=chPAD)
        tl_index = torch.nn.utils.rnn.pad_sequence(tl_index, batch_first=True, padding_value=tlPAD)
        return ch_index, tl_index


class TestDataset(torch.utils.data.Dataset):
    @staticmethod
    def get_data(dataset_path, nums=None):
        datas = pd.read_pickle(dataset_path, compression='gzip')
        id_list = list(datas["id"])
        ch_data = list(datas["ch"])
        if nums is None:
            return id_list, ch_data
        else:
            return id_list[:nums], ch_data[:nums]

    def __init__(self, dataset_path, ch_tokenizer, nums=None):
        id_list, ch_data = self.get_data(dataset_path, nums=nums)
        self.id_list = id_list
        self.ch_data = ch_data
        self.ch_tokenizer = ch_tokenizer

    def __getitem__(self, index):
        ch = self.ch_data[index]
        ch_index = self.ch_tokenizer.encode(sentence=ch)
        return ch_index

    def __len__(self):
        return len(self.ch_data)

    def find_idlist(self):
        return self.id_list

    def collate_fn(self, batch_list):
        chBOS = self.ch_tokenizer.BOS
        chEOS = self.ch_tokenizer.EOS
        ch_index = []
        for ch in batch_list:
            # TODO
            ch_index.append(torch.tensor([chBOS] + ch + [chEOS]))
            # ch_index.append(torch.tensor(ch))
        ch_index = torch.stack(ch_index)
        return ch_index