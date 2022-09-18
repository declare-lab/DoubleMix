import torch
import fairseq
import pickle
import os
import json
from tqdm.notebook import trange, tqdm

def translate_de(start, end, file_name, temperature=0.9):
    trans_a = {}
    trans_b = {}
    for idx in tqdm(range(start, end)):
        trans_a[train_idxs[idx]] = de2en.translate(en2de.translate(train_text_a[idx],  sampling = True, temperature = temperature),  sampling = True, temperature = 0.9)
        trans_b[train_idxs[idx]] = de2en.translate(en2de.translate(train_text_b[idx],  sampling = True, temperature = temperature),  sampling = True, temperature = 0.9)
        if idx % 500 == 0:
            with open("../../dataset/multinli/"+file_name, 'wb') as f:
                pickle.dump((trans_a, trans_b), f)
    with open("../../dataset/multinli/"+file_name, 'wb') as f:
        pickle.dump((trans_a, trans_b), f)

def translate_ru(start, end, file_name, temperature=0.9):
    trans_a = {}
    trans_b = {}
    for idx in tqdm(range(start, end)):
        trans_a[train_idxs[idx]] = ru2en.translate(en2ru.translate(train_text_a[idx],  sampling = True, temperature = temperature),  sampling = True, temperature = 0.9)
        trans_b[train_idxs[idx]] = ru2en.translate(en2ru.translate(train_text_b[idx],  sampling = True, temperature = temperature),  sampling = True, temperature = 0.9)
        if idx % 500 == 0:
            with open("../../dataset/multinli/"+file_name, 'wb') as f:
                pickle.dump((trans_a, trans_b), f)

    with open("../../dataset/multinli/"+file_name, 'wb') as f:
        pickle.dump((trans_a, trans_b), f)


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

    with open("../../dataset/multinli/train.json") as f:
            train_data = json.load(f)

    train_labels = [v['gold_label'] for v in train_data]
    train_text_a = [v['sentence1'] for v in train_data]
    train_text_b = [v['sentence2'] for v in train_data]
    #data_idxs = [v['pairID'] for v in train_data]
    data_idxs = [str(i) for i,v in enumerate(train_data)]
    print('train data length:', len(data_idxs))

    # some pairIDs are the same, so we add a 'new' string to make the ID key sole
    train_idxs = data_idxs
   # for idx in data_idxs:
    #    if idx not in train_idxs:
     #       train_idxs.append(idx)
      #  else:
       #     print(idx)
        #    train_idxs.append(idx+'new')
    assert len(train_idxs) == len(data_idxs)

    translate_de(0,len(train_idxs),'multinli_de.pkl')
    translate_ru(0,len(train_idxs),'multinli_ru.pkl')
