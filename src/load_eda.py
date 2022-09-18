import pickle 
import pandas as pd
import numpy as np
import json
import random
from datasets import load_dataset  
from read_data import label_map
from read_data import loader as loader_nli
from load_tcdata import loader

def get_eda(dataset="trec", aug=0, rand=False):
    # Load dataset
    if dataset == "sst2":
        if rand:
            with open("../dataset/"+dataset+"/sst2_eda.json", "r") as f:
                train_data = json.load(f)
        else:
            with open("../dataset/"+dataset+"/sst2_sr.json", "r") as f:
                train_data = json.load(f)
        dev_data = pd.read_csv('../dataset/sst2/dev.tsv', sep='\t')
        dev_data.columns = ['text','label']

        test_data = pd.read_csv('../dataset/sst2/test.tsv', sep='\t')
        test_data.columns = ['text','label']
        
    elif dataset == "puns":
        if rand:
            with open("../dataset/"+dataset+"/puns_eda.json", "r") as f:
                train_data = json.load(f)
        else:
            with open("../dataset/"+dataset+"/puns_sr.json", "r") as f:
                train_data = json.load(f)
        dev_data = pd.read_csv('../dataset/puns/dev.tsv')
        test_data = pd.read_csv('../dataset/puns/test.tsv')
    elif dataset == "multinli" or dataset == "snli":
        with open("../dataset/"+dataset+"/"+dataset+"_sr.json", "r") as f1, \
        open("../dataset/"+dataset+"/dev.json", "r") as f2, \
        open("../dataset/"+dataset+"/test.json", "r") as f3:
                train_data = json.load(f1)
                dev_data = json.load(f2)
                test_data = json.load(f3)    
        for v in train_data[:]:
            if v["label"] == "-":
                train_data.remove(v)
        for v in dev_data[:]:
            if v["gold_label"] == "-":
                dev_data.remove(v)
        for v in test_data[:]:
            if v["gold_label"] == "-":
                test_data.remove(v) 
    else:
        if dataset == "imdb" or dataset == "trec":
            test_size = 0.1
        else:
            raise NotImplementedError
        ds = load_dataset(dataset, split='train')
        test_data = load_dataset(dataset, split='test')
        _, dev_data = ds.train_test_split(test_size=test_size, seed=42).values()
        if rand:
            with open("../dataset/"+dataset+"/"+dataset+"_eda.json", "r") as f:
                train_data = json.load(f)
        else:
            with open("../dataset/"+dataset+"/"+dataset+"_sr.json", "r") as f:
                train_data = json.load(f)

    # transfer to np.array
    if dataset == "trec":
        train_labels = np.array([v["label"] for v in train_data])
        dev_labels = np.array(dev_data['label-coarse'])
        test_labels = np.array(test_data['label-coarse'])
    elif dataset == "sst2" or dataset == "imdb" or dataset == "puns":
        train_labels = np.array([v["label"] for v in train_data])
        dev_labels = np.array(dev_data['label'])
        test_labels = np.array(test_data['label'])
    elif dataset == "multinli" or dataset == "snli":
        train_labels = np.array([label_map(v["label"]) for v in train_data])
        dev_labels = np.array([label_map(v["gold_label"]) for v in dev_data])
        test_labels = np.array([label_map(v["gold_label"]) for v in test_data])
    else:
        train_labels = np.array(train_data['label'])
        dev_labels = np.array(dev_data['label'])
        test_labels = np.array(test_data['label'])

    if dataset == "multinli" or dataset == "snli":
        train_text_a = np.array([v["sentence1"] for v in train_data])
        train_text_b = np.array([v["sentence2"] for v in train_data])
        dev_text_a = np.array([v["sentence1"] for v in dev_data])
        dev_text_b = np.array([v["sentence2"] for v in dev_data])
        test_text_a = np.array([v["sentence1"] for v in test_data])
        test_text_b = np.array([v["sentence2"] for v in test_data])
    else:
        train_text = np.array([v["sentence"] for v in train_data])
        dev_text = np.array(dev_data['text'])
        test_text = np.array(test_data['text'])
    
    if dataset == "multinli" or dataset == "snli":
        sr11 = np.array([v["sentence1-sr1"] for v in train_data])
        sr12 = np.array([v["sentence1-sr2"] for v in train_data])
        sr21 = np.array([v["sentence2-sr1"] for v in train_data])
        sr22 = np.array([v["sentence2-sr2"] for v in train_data])
    else:
        if rand:
            sr = np.array([v["sr"] for v in train_data])
            ri = np.array([v["ri"] for v in train_data])
            rs = np.array([v["rs"] for v in train_data])
            rd = np.array([v["rd"] for v in train_data])
        else:    
            sr1 = np.array([v["sr1"] for v in train_data])
            sr2 = np.array([v["sr2"] for v in train_data])
    
    # Loader
    if aug:
        if rand:           
            if dataset != "multinli" and dataset != "snli":
                with open("../dataset/"+dataset+"/"+dataset+"_de.pkl", 'rb') as f1, open("../dataset/"+dataset+"/"+dataset+"_ru.pkl", 'rb') as f2:
                    de_data = pickle.load(f1)
                    ru_data = pickle.load(f2)
                
                text_de = list(de_data.values())
                assert len(text_de) == len(train_data)
                text_ru = list(ru_data.values())
                assert len(text_ru) == len(train_data)
                de = np.array(text_de)
                ru = np.array(text_ru)
                idx1 = random.randint(0,1)
                idx2 = random.randint(0,3)
                aug1 = [de,ru][idx1]
                aug2 = [sr, ri, rs, rd][idx2]
        else:
            if dataset == "multinli" or dataset == "snli":
                aug_a1 = sr11
                aug_b1 = sr21
                aug_a2 = sr12
                aug_b2 = sr22
            else:
                aug1 = sr1
                aug2 = sr2
        if dataset == "multinli" or dataset == "snli":
            train_dataset = loader_nli(train_text_a, train_text_b, train_labels, aug_a1=aug_a1, aug_b1=aug_b1, aug_a2=aug_a2, aug_b2=aug_b2)
            dev_dataset = loader_nli(dev_text_a, dev_text_b, dev_labels)
            test_dataset = loader_nli(test_text_a, test_text_b, test_labels)
        else:
            train_dataset = loader(train_text, train_labels, aug_1=aug1, aug_2=aug2)
            dev_dataset = loader(dev_text, dev_labels)
            test_dataset = loader(test_text, test_labels)
    else:
        if dataset == "multinli" or dataset == "snli":
            # train_text_a = np.append(train_text_a, sr11)
            train_text_a = np.append(train_text_a, sr12)
            # train_text_b = np.append(train_text_b, sr21)
            train_text_b = np.append(train_text_b, sr22)
            # train_labels1 = np.append(train_labels, train_labels)
            # train_labels = np.append(train_labels1, train_labels)
            train_labels = np.append(train_labels, train_labels)
            train_dataset = loader_nli(train_text_a, train_text_b, train_labels)
            dev_dataset = loader_nli(dev_text_a, dev_text_b, dev_labels)
            test_dataset = loader_nli(test_text_a, test_text_b, test_labels)
        else:
            train_text = np.append(train_text, sr1)
            # train_text = np.append(train_text, sr2)
            train_labels = np.append(train_labels, train_labels)
            train_dataset = loader(train_text, train_labels)
            dev_dataset = loader(dev_text, dev_labels)
            test_dataset = loader(test_text, test_labels)

    print("train data size:", len(train_labels))
    print("Get data: done!")
    return train_dataset, dev_dataset, test_dataset