import torch
import fairseq
import pickle
import os
from tqdm.notebook import trange, tqdm
import pandas as pd

def translate_de(start, end, file_name,temperature=0.9):
    trans = {}
    for idx in tqdm(range(start, end)):
        trans[train_idxs[idx]] = de2en.translate(en2de.translate(train_text[idx],  sampling = True, temperature = temperature),  sampling = True, temperature = 0.9)
        if idx % 500 == 0:
            with open(file_name, 'wb') as f:
                pickle.dump(trans, f)
    with open(file_name, 'wb') as f:
        pickle.dump(trans, f)

def translate_ru(start, end, file_name,temperature=0.9):
    trans = {}
    for idx in tqdm(range(start, end)):
        trans[train_idxs[idx]] = ru2en.translate(en2ru.translate(train_text[idx],  sampling = True, temperature = temperature),  sampling = True, temperature = 0.9)
        if idx % 500 == 0:
            with open(file_name, 'wb') as f:
                pickle.dump(trans, f)
    with open(file_name, 'wb') as f:
        pickle.dump(trans, f)

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("gpu num: ", n_gpu)

    en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
    de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

    en2de = en2de.cuda()
    de2en = de2en.cuda()

    en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
    ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')

    en2ru.cuda()
    ru2en.cuda()

    train_data = pd.read_csv('train.tsv', sep='\t')
    
    train_labels = train_data['label']
    train_text = train_data['sentence']
    train_idxs = [str(i) for i in range(len(train_data))]
    print('train data length:', len(train_idxs))

    translate_de(0,len(train_idxs),'sst2_de.pkl')
