import json
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import random

def label_map(label):
    label_dict = {"entailment": 0, "neutral": 1, "contradiction": 2}
    if label not in label_dict.keys():
        print("label is not in dict!", label)        
    return label_dict[label]
        

class loader(Dataset):
    def __init__(self, seq_a, seq_b, labels, aug_a1=None, aug_b1=None, aug_a2=None, aug_b2=None, max_seq_len=256):
        self.seq_a = seq_a
        self.seq_b = seq_b
        self.labels = labels
        self.aug_a1 = aug_a1
        self.aug_b1 = aug_b1
        self.aug_a2 = aug_a2
        self.aug_b2 = aug_b2
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text_a = self.seq_a[idx][:self.max_seq_len]
        text_b = self.seq_b[idx][:self.max_seq_len]

        if self.aug_a1 is not None:
            trans_a1 = self.aug_a1[idx]
            trans_b1 = self.aug_b1[idx]
            
            trans_a2 = self.aug_a2[idx]
            trans_b2 = self.aug_b2[idx]

            return str(text_a), str(text_b), str(trans_a1), str(trans_b1), \
                str(trans_a2), str(trans_b2), self.labels[idx]

        return str(text_a), str(text_b), self.labels[idx]

def collate_fn(batch, aug, tokenizer):
    processed_batch = {}
    batch_size = len(batch)
    if aug:
        sents_a, sents_b = [], []
        trans_a1, trans_b1 = [], []
        trans_a2, trans_b2 = [], []
        for sample in batch:
            sents_a.append(sample[0])
            sents_b.append(sample[1])
            trans_a1.append(sample[2])
            trans_b1.append(sample[3])
            trans_a2.append(sample[4])
            trans_b2.append(sample[5])    

        max_length = tokenizer(sents_a, sents_b, padding='longest', return_tensors='pt')['input_ids'].size(1)    
        batch_token = tokenizer(sum([sents_a, trans_a1, trans_a2], []), sum([sents_b, trans_b1, trans_b2], []),
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
        processed_batch['labels'] = torch.tensor([sample[6] for sample in batch])
    else: 
        sents_a = [sample[0] for sample in batch]
        sents_b = [sample[1] for sample in batch]
        processed_batch['inputs'] = tokenizer(sents_a, sents_b, padding='longest', return_tensors='pt', return_token_type_ids=True)
        processed_batch['labels'] = torch.tensor([sample[2] for sample in batch])
    return processed_batch


def get_data(data_path, dataset="snli", aug=0, size=1000):
    
    if size == 1000:
        suffix = "_1k"
    elif size == 2500:
        suffix = "_2k5"
    elif size == 5000:
        suffix = "_5k"
    elif size == 10000:
        suffix = "_1w" 
    elif size == 50000:
        suffix = "_5w"
    else:
        suffix = ""  
    
    with open(data_path+dataset+"/train"+suffix+".json") as f1, open(data_path+dataset+"/dev.json") as f2, open(data_path+dataset+"/test.json") as f3:
        train_data = json.load(f1)
        dev_data = json.load(f2)
        test_data = json.load(f3)
    
    if aug:
        # applying back-translation augmentation
        with open(data_path+dataset+"/"+dataset+"_de"+suffix+".pkl", 'rb') as f1, open(data_path+dataset+"/"+dataset+"_ru"+suffix+".pkl", 'rb') as f2:
            de_data = pickle.load(f1)
            ru_data = pickle.load(f2)
        
        de_a = list(de_data[0].values())
        de_b = list(de_data[1].values())
        assert len(de_a) == len(de_b) == len(train_data)
        ru_a = list(ru_data[0].values())
        ru_b = list(ru_data[1].values())
        assert len(ru_a) == len(ru_b) == len(train_data)
        
        for i,v in enumerate(train_data):
            v['sentence1_de'] = de_a[i]
            v['sentence2_de'] = de_b[i]
            v['sentence1_ru'] = ru_a[i]
            v['sentence2_ru'] = ru_b[i]
        
    print("train data size:", len(train_data))
    # Remove the samples where the gold label is '-'
    for v in train_data[:]:
        if v["gold_label"] == "-":
            train_data.remove(v)
    for v in dev_data[:]:
        if v["gold_label"] == "-":
            dev_data.remove(v)
    for v in test_data[:]:
        if v["gold_label"] == "-":
            test_data.remove(v)
            
    train_text_a = np.array([v["sentence1"] for v in train_data])
    train_text_b = np.array([v["sentence2"] for v in train_data])
    train_labels = np.array([label_map(v["gold_label"]) for v in train_data])


    dev_text_a = np.array([v["sentence1"] for v in dev_data])
    dev_text_b = np.array([v["sentence2"] for v in dev_data])
    dev_labels = np.array([label_map(v["gold_label"]) for v in dev_data])

    test_text_a = np.array([v["sentence1"] for v in test_data])
    test_text_b = np.array([v["sentence2"] for v in test_data])
    test_labels = np.array([label_map(v["gold_label"]) for v in test_data])

    if aug:
        de_text_a = np.array([v["sentence1_de"] for v in train_data])
        de_text_b = np.array([v["sentence2_de"] for v in train_data])
        ru_text_a = np.array([v["sentence1_ru"] for v in train_data])
        ru_text_b = np.array([v["sentence2_ru"] for v in train_data])
        train_dataset = loader(train_text_a, train_text_b, train_labels, aug_a1=de_text_a, aug_b1=de_text_b, aug_a2=ru_text_a, aug_b2=ru_text_b)
    else:
        train_dataset = loader(train_text_a, train_text_b, train_labels)
    dev_dataset = loader(dev_text_a, dev_text_b, dev_labels)
    test_dataset = loader(test_text_a, test_text_b, test_labels)
    print("Get data: done!")
    return train_dataset, dev_dataset, test_dataset

def get_nli_rand(dataset="snli"):
    # Load dataset
    with open("../dataset/"+dataset+"/"+dataset+"_sr.json", "r") as f1, \
        open("../dataset/"+dataset+"/dev.json", "r") as f2, \
        open("../dataset/"+dataset+"/test.json", "r") as f3:
                train_data = json.load(f1)
                dev_data = json.load(f2)
                test_data = json.load(f3)

    # Load Back-trans samples
    with open("../dataset/"+dataset+"/"+dataset+"_de.pkl", 'rb') as f1, open("../dataset/"+dataset+"/"+dataset+"_ru.pkl", 'rb') as f2:
        de_data = pickle.load(f1)
        ru_data = pickle.load(f2)
        
    de_a = list(de_data[0].values())
    de_b = list(de_data[1].values())
    assert len(de_a) == len(de_b) == len(train_data)
    ru_a = list(ru_data[0].values())
    ru_b = list(ru_data[1].values())
    assert len(ru_a) == len(ru_b) == len(train_data)

    for i,v in enumerate(train_data):
        v['sentence1_de'] = de_a[i]
        v['sentence2_de'] = de_b[i]
        v['sentence1_ru'] = ru_a[i]
        v['sentence2_ru'] = ru_b[i]

    # Remove the samples where the gold label is '-'
    for v in train_data[:]:
        if v["label"] == "-":
            train_data.remove(v)
    for v in dev_data[:]:
        if v["gold_label"] == "-":
            dev_data.remove(v)
    for v in test_data[:]:
        if v["gold_label"] == "-":
            test_data.remove(v)

    # transfer to np.array
    train_text_a = np.array([v["sentence1"] for v in train_data])
    train_text_b = np.array([v["sentence2"] for v in train_data])

    dev_text_a = np.array([v["sentence1"] for v in dev_data])
    dev_text_b = np.array([v["sentence2"] for v in dev_data])

    test_text_a = np.array([v["sentence1"] for v in test_data])
    test_text_b = np.array([v["sentence2"] for v in test_data])

    train_labels = np.array([label_map(v["label"]) for v in train_data])
    dev_labels = np.array([label_map(v["gold_label"]) for v in dev_data])
    test_labels = np.array([label_map(v["gold_label"]) for v in test_data])

    # Load sr samples
    sr11 = np.array([v["sentence1-sr1"] for v in train_data])
    sr12 = np.array([v["sentence1-sr2"] for v in train_data])
    sr21 = np.array([v["sentence2-sr1"] for v in train_data])
    sr22 = np.array([v["sentence2-sr2"] for v in train_data])

    # Back-trans samples
    de_text_a = np.array([v["sentence1_de"] for v in train_data])
    de_text_b = np.array([v["sentence2_de"] for v in train_data])
    ru_text_a = np.array([v["sentence1_ru"] for v in train_data])
    ru_text_b = np.array([v["sentence2_ru"] for v in train_data])

    # Randomly select augs
    idx = random.randint(0,1)
    aug_a1 = [de_text_a, ru_text_a][idx]
    aug_b1 = [de_text_b, ru_text_b][idx]
    aug_a2 = [sr11, sr12][idx]
    aug_b2 = [sr21, sr22][idx]

    # Loader
    train_dataset = loader(train_text_a, train_text_b, train_labels, aug_a1=aug_a1, aug_b1=aug_b1, aug_a2=aug_a2, aug_b2=aug_b2)
    dev_dataset = loader(dev_text_a, dev_text_b, dev_labels)
    test_dataset = loader(test_text_a, test_text_b, test_labels)

    print("train data size:", len(train_labels))
    print("Get data: done!")
    return train_dataset, dev_dataset, test_dataset