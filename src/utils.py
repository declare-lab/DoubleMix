import numpy as np
from tqdm import tqdm

num_label = {
        "sst2":2,
        "trec":6,
        "imdb":2,
        "ag_news":4,
        "snli":3,
        "multinli":3,
        "puns":2,
    }

def mix_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def vis_aug(train_loader, model, args):
    model.eval()
    batch_embeddings = []
    batch_targets = []
    pbar = tqdm(train_loader)
    for batch_idx, batch in enumerate(pbar):
        orig_inputs, aug_inputs1, aug_inputs2, targets = batch['orig_inputs'], batch['aug_inputs1'], \
        batch['aug_inputs2'], batch['labels']
        for k, v in orig_inputs.items(): orig_inputs[k] = v.cuda()
        for k, v in aug_inputs1.items(): aug_inputs1[k] = v.cuda()
        for k, v in aug_inputs2.items(): aug_inputs2[k] = v.cuda()
    
        mix_layer = np.random.choice(args.mix_layer_set, 1)[0] - 1
        if (args.alpha)<1e-6:
            lam=1
        else:
            if args.beta==-1:                        
                lam = np.random.beta(args.alpha, args.alpha)
            else:
                lam = np.random.beta(args.alpha, args.beta)
            lam = max(lam, 1-lam)
        ws = np.random.dirichlet([args.tau] * args.n_sample)
        _, (pooled_output, _) = model(orig_inputs, aug_inputs1, aug_inputs2, lam=lam, ws=ws, mix_layer=mix_layer)

        batch_embeddings.append(pooled_output.detach().cpu().numpy())
        batch_targets.append(targets.detach().cpu().numpy())

        with open('model_logs/embeddings_aug.npy', 'wb') as f:
            all_embedding = np.concatenate(batch_embeddings, axis=0)
            np.save(f, all_embedding)

        with open('model_logs/targets_aug.npy', 'wb') as f:
            all_targets = np.concatenate(batch_targets, axis=0)
            np.save(f, all_targets)
        pbar.set_description('batch idx={:4f}'.format(batch_idx))

def vis(train_loader, model):
    model.eval()
    pbar = tqdm(train_loader)
    batch_embeddings = []
    batch_targets = []
    
    for batch_idx, batch in enumerate(pbar):
        inputs, targets = batch['inputs'], batch['labels']
        targets = targets.cuda(non_blocking=True)     
        for k, v in inputs.items(): inputs[k] = v.cuda()
        
        _, (pooled_output, _) = model(inputs)
        batch_embeddings.append(pooled_output.detach().cpu().numpy())
        batch_targets.append(targets.detach().cpu().numpy())

        with open('model_logs/embeddings.npy', 'wb') as f:
            all_embedding = np.concatenate(batch_embeddings, axis=0)
            np.save(f, all_embedding)

        with open('model_logs/targets.npy', 'wb') as f:
            all_targets = np.concatenate(batch_targets, axis=0)
            np.save(f, all_targets)
        pbar.set_description('batch idx={:.4f}'.format(batch_idx))