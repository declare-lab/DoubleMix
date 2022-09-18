
from eda import *
import pandas as pd
import json
import numpy as np
from datasets import load_dataset

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--data_path", required=False, type=str, default="../../dataset/", help="input file of unaugmented data")
# ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
ap.add_argument("--dataset", required=False, type=str, default="snli", help="augmented dataset")
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha_sr", required=False, type=float, help="percent of words in each sentence to be replaced by synonyms")
ap.add_argument("--alpha_ri", required=False, type=float, help="percent of words in each sentence to be inserted")
ap.add_argument("--alpha_rs", required=False, type=float, help="percent of words in each sentence to be swapped")
ap.add_argument("--alpha_rd", required=False, type=float, help="percent of words in each sentence to be deleted")
args = ap.parse_args()

#number of augmented sentences to generate per original sentence
num_aug = 4 #default
if args.num_aug:
    num_aug = args.num_aug

#how much to replace each word by synonyms
alpha_sr = 0.1#default
if args.alpha_sr is not None:
    alpha_sr = args.alpha_sr

#how much to insert new words that are synonyms
alpha_ri = 0.1#default
if args.alpha_ri is not None:
    alpha_ri = args.alpha_ri

#how much to swap words
alpha_rs = 0.1#default
if args.alpha_rs is not None:
    alpha_rs = args.alpha_rs

#how much to delete words
alpha_rd = 0.1#default
if args.alpha_rd is not None:
    alpha_rd = args.alpha_rd

if alpha_sr == alpha_ri == alpha_rs == alpha_rd == 0:
     ap.error('At least one alpha should be greater than zero')

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
#generate more data with standard augmentation
def gen_sst2(train_orig, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):
    lines = pd.read_csv(train_orig, sep='\t')
    items = []
    for i in range(len(lines)):
        label = lines["label"][i]
        sentence = lines["sentence"][i]
        aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
        item = {}
        item["label"] = label
        item["sentence"] = sentence
        item["sr"] = aug_sentences[0]
        item["ri"] = aug_sentences[1]
        item["rs"] = aug_sentences[2]
        item["rd"] = aug_sentences[3]
        items.append(item)
    with open(args.data_path+args.dataset+"/"+args.dataset+"_eda.json", "w") as f:
        json.dump(items, f, sort_keys=False, indent=4, separators=(',', ': '), cls=NpEncoder)

def gen_trec(train_orig, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):
    items = []
    
    for v in train_orig:
        label = v["label-coarse"]
        sentence = v["text"]
        aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
        item = {}
        item["label"] = label
        item["sentence"] = sentence
        item["sr"] = aug_sentences[0]
        item["ri"] = aug_sentences[1]
        item["rs"] = aug_sentences[2]
        item["rd"] = aug_sentences[3]
        items.append(item)
    with open(args.data_path+args.dataset+"/"+args.dataset+"_eda.json", "w") as f:
        json.dump(items, f, sort_keys=False, indent=4, separators=(',', ': '), cls=NpEncoder)

def gen_nli(inputfile, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):
    with open(inputfile, "r") as f:
        train_data = json.load(f)
    items = []
    for v in train_data:
        label = v["gold_label"]
        sentence1 = v["sentence1"]
        sentence2 = v["sentence2"]
        aug_sentences1 = eda(sentence1, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
        aug_sentences2 = eda(sentence2, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
        item = {}
        item["label"] = label
        item["sentence1"] = sentence1
        item["sentence2"] = sentence2
        item["sentence1-sr"] = aug_sentences1[0]
        item["sentence1-ri"] = aug_sentences1[1]
        item["sentence1-rs"] = aug_sentences1[2]
        item["sentence1-rd"] = aug_sentences1[3]
        item["sentence2-sr"] = aug_sentences2[0]
        item["sentence2-ri"] = aug_sentences2[1]
        item["sentence2-rs"] = aug_sentences2[2]
        item["sentence2-rd"] = aug_sentences2[3]
        items.append(item)
    with open(args.data_path+args.dataset+"/"+args.dataset+"_eda.json", "w") as f:
        json.dump(items, f, sort_keys=False, indent=4, separators=(',', ': '), cls=NpEncoder)

def gen_imdb(train_orig, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):
    items = [] 
    for v in train_orig:
        label = v["label"]
        sentence = v["text"]
        aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
        item = {}
        item["label"] = label
        item["sentence"] = sentence
        item["sr"] = aug_sentences[0]
        item["ri"] = aug_sentences[1]
        item["rs"] = aug_sentences[2]
        item["rd"] = aug_sentences[3]
        items.append(item)
    with open(args.data_path+args.dataset+"/"+args.dataset+"_eda.json", "w") as f:
        json.dump(items, f, sort_keys=False, indent=4, separators=(',', ': '), cls=NpEncoder)

def gen_agnews(train_orig, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):
    items = [] 
    for v in train_orig:
        label = v["label"]
        sentence = v["text"]
        aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
        item = {}
        item["label"] = label
        item["sentence"] = sentence
        item["sr"] = aug_sentences[0]
        item["ri"] = aug_sentences[1]
        item["rs"] = aug_sentences[2]
        item["rd"] = aug_sentences[3]
        items.append(item)
    with open(args.data_path+args.dataset+"/"+args.dataset+"_eda.json", "w") as f:
        json.dump(items, f, sort_keys=False, indent=4, separators=(',', ': '), cls=NpEncoder)

def gen_puns(train_orig, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):
    lines = pd.read_csv(train_orig)
    items = []
    for i in range(len(lines)):
        label = lines["label"][i]
        sentence = lines["text"][i]
        aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
        item = {}
        item["label"] = label
        item["sentence"] = sentence
        item["sr"] = aug_sentences[0]
        item["ri"] = aug_sentences[1]
        item["rs"] = aug_sentences[2]
        item["rd"] = aug_sentences[3]
        items.append(item)
    with open(args.data_path+args.dataset+"/"+args.dataset+"_eda.json", "w") as f:
        json.dump(items, f, sort_keys=False, indent=4, separators=(',', ': '), cls=NpEncoder)

#main function
if __name__ == "__main__":
    if args.dataset == "sst2":
        inputfile = args.data_path+args.dataset+"/train.tsv"
        gen_sst2(inputfile, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug)
    elif args.dataset == "trec":
        data = load_dataset('trec', split='train')
        train_data, _ = data.train_test_split(test_size=0.1, seed=42).values()
        gen_trec(train_data, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug)
    elif args.dataset == "snli" or args.dataset == "multinli":
        inputfile = args.data_path+args.dataset+"/train.json"
        gen_nli(inputfile, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug)
    elif args.dataset == "imdb":
        data = load_dataset('imdb', split='train')
        train_data, _ = data.train_test_split(test_size=0.1, seed=42).values()
        gen_imdb(train_data, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug)
    elif args.dataset == "ag_news":
        data = load_dataset('ag_news', split='train')
        train_data, _ = data.train_test_split(test_size=0.05, seed=42).values()
        gen_agnews(train_data, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug)
    elif args.dataset == "puns":
        inputfile = args.data_path+args.dataset+"/train.tsv"
        gen_puns(inputfile, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug)
    else:
        raise NotImplementedError
