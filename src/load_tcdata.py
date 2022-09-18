import pickle
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset  
from torch.utils.data import Dataset

class loader(Dataset):
    def __init__(self, seq, labels, aug_1=None, aug_2=None, max_seq_len=256):
        self.seq = seq
        self.labels = labels        
        self.aug_1 = aug_1
        self.aug_2 = aug_2
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.seq[idx][:self.max_seq_len]

        if self.aug_1 is not None:
            result1 = self.aug_1[idx]
            result2 = self.aug_2[idx]
            return str(text), str(result1), str(result2), self.labels[idx]

        return str(text), self.labels[idx]

def collate_fn(batch, aug, tokenizer):
    processed_batch = {}
    batch_size = len(batch)
    if aug:
        sents = []
        result1, result2 = [], []
        for sample in batch:
            sents.append(sample[0])
            result1.append(sample[1])
            result2.append(sample[2])      
        # batch_token = tokenizer(sum([sents, result1, result2], []), padding='longest', return_tensors='pt', return_token_type_ids=True)
        max_length = tokenizer(sents, padding='longest', return_tensors='pt')['input_ids'].size(1)    
        batch_token = tokenizer(sum([sents, result1, result2], []),
                                max_length=max_length, truncation=True, padding='max_length', return_tensors='pt', return_token_type_ids=True)
        # retokenize
        processed_batch['orig_inputs'] = {'input_ids': batch_token['input_ids'][:batch_size],
                                        'attention_mask': batch_token['attention_mask'][:batch_size],
                                        'token_type_ids': batch_token['token_type_ids'][:batch_size]}
        processed_batch['aug_inputs1'] = {'input_ids': batch_token['input_ids'][batch_size:2*batch_size],
                                        'attention_mask': batch_token['attention_mask'][batch_size:2*batch_size],
                                        'token_type_ids': batch_token['token_type_ids'][batch_size:2*batch_size]}
        processed_batch['aug_inputs2'] = {'input_ids': batch_token['input_ids'][2*batch_size:],
                                        'attention_mask': batch_token['attention_mask'][2*batch_size:],
                                        'token_type_ids': batch_token['token_type_ids'][2*batch_size:]}
        processed_batch['labels'] = torch.tensor([sample[3] for sample in batch])
    else: 
        sents = [sample[0] for sample in batch]
        processed_batch['inputs'] = tokenizer(sents, padding='longest', return_tensors='pt', return_token_type_ids=True)
        processed_batch['labels'] = torch.tensor([sample[1] for sample in batch])
    return processed_batch

def get_tcdata(dataset="sst2", aug=0):

    if dataset == "sst2":
        train_data = pd.read_csv('../dataset/sst2/train.tsv', sep='\t')
        dev_data = pd.read_csv('../dataset/sst2/dev.tsv', sep='\t')

        test_data = pd.read_csv('../dataset/sst2/test.tsv', sep='\t')
        test_data.columns = ['text','label']
        train_data.columns = ['text','label']
        dev_data.columns = ['text','label']
        
    elif dataset == "puns":
        train_data = pd.read_csv('../dataset/puns/train.tsv')
        dev_data = pd.read_csv('../dataset/puns/dev.tsv')
        test_data = pd.read_csv('../dataset/puns/test.tsv')
    else:
        if dataset == "imdb" or "trec":
            test_size = 0.1
        ds = load_dataset(dataset, split='train')
        train_data, dev_data = ds.train_test_split(test_size=test_size, seed=42).values()

        test_data = load_dataset(dataset, split='test')
        print("test data size:", len(test_data))

    if aug:
        with open("../dataset/"+dataset+"/"+dataset+"_de.pkl", 'rb') as f1, open("../dataset/"+dataset+"/"+dataset+"_ru.pkl", 'rb') as f2:
            de_data = pickle.load(f1)
            ru_data = pickle.load(f2)
        
        text_de = list(de_data.values())
        assert len(text_de) == len(train_data)
        text_ru = list(ru_data.values())
        assert len(text_ru) == len(train_data)
        de = np.array(text_de)
        ru = np.array(text_ru)
        
    if dataset == "trec":
        train_labels = np.array(train_data['label-coarse'])
        dev_labels = np.array(dev_data['label-coarse'])
        test_labels = np.array(test_data['label-coarse'])
    else:
        train_labels = np.array(train_data['label'])
        dev_labels = np.array(dev_data['label'])
        test_labels = np.array(test_data['label'])
    train_text = np.array(train_data['text'])
    dev_text = np.array(dev_data['text'])
    test_text = np.array(test_data['text'])

    if aug: 
        train_dataset = loader(train_text, train_labels, aug_1=de, aug_2=ru)
    else:
        train_dataset = loader(train_text, train_labels)
    print("train data size:", len(train_labels))
    dev_dataset = loader(dev_text, dev_labels)
    test_dataset = loader(test_text, test_labels)
    print("Get data: done!")
    return train_dataset, dev_dataset, test_dataset