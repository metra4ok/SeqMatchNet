import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import numpy as np
import time
import wandb

from utils import seq2Batch, getRecallAtN, computeMatches, evaluate, N_VALUES


def export_predictions(preds, eval_set, filename='test_2014-12-16-18-44-24'):
    print("\n====> Exporting predictions")
    print(f"\tpredictions.shape = {preds.shape}")
    predictions = []
    for q_idx, pred in enumerate(preds):
        top_preds = []
        for idx in pred:
            indices = eval_set.getIndices(idx)
            imgnames = eval_set.dbStruct.dbImage[indices[0]:indices[-1]+1]
            imgnames = ['/'.join(n.split('/')[:-2])+'/'+n.split('/')[-1] for n in imgnames]
            top_preds.append(imgnames)
        q_indices = eval_set.getIndices(q_idx)
        q_imgnames = eval_set.dbStruct.qImage[q_indices[0]:q_indices[-1]+1]
        q_imgnames = ['/'.join(n.split('/')[:-2])+'/'+n.split('/')[-1] for n in q_imgnames]
        pair = (q_imgnames, top_preds)
        predictions.append(pair)
    save_to = f'/home/docker_seqpntr/SeqPNTR/results/seqmatchnet_predictions_{filename}.pkl'
    with open(save_to, 'wb') as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"====> Dumped results to file: {save_to}")
    print(f"\n\nsample: \n\tquery:\n{predictions[0][0]}\n\tpreds:\n{predictions[0][1]}\n\n")


def test(opt, model, encoder_dim, device, eval_set, writer, epoch=0, extract_noEval=False):
    # TODO what if features dont fit in memory? 
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=False)

    model.eval()
    with torch.no_grad():
        print('====> Extracting Features')
        pool_size = encoder_dim
        validInds = eval_set.validInds
        dbFeat_single = torch.zeros((len(eval_set), pool_size),device=device)
        durs_batch = []
        for iteration, (input, indices) in tqdm(enumerate(test_data_loader, 1),total=len(test_data_loader)-1, leave=False):
            t1 = time.time()
            image_encoding = seq2Batch(input).float().to(device)
            global_single_descs = model.pool(image_encoding).squeeze()
            dbFeat_single[indices] = global_single_descs

            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, 
                    len(test_data_loader)), flush=True)
            durs_batch.append(time.time()-t1)
        del input, image_encoding, global_single_descs

    del test_data_loader
    print("Average batch time:", np.mean(durs_batch), np.std(durs_batch))

    outSeqL = opt.seqL
    # use the original single descriptors for fast seqmatch over dMat (MSLS-like non-continuous dataset not supported)
    if (not opt.pooling and opt.matcher is None) and ('nordland' in opt.dataset.lower() or 'tmr' in opt.dataset.lower() or 'ox' in opt.dataset.lower()):
        dbFeat = dbFeat_single
        numDb = eval_set.numDb_full
    # fill sequences centered at single images
    else:
        dbFeat = torch.zeros(len(validInds), outSeqL, pool_size, device=device)
        numDb = eval_set.dbStruct.numDb
        for ind in range(len(validInds)):
            dbFeat[ind] = dbFeat_single[eval_set.getSeqIndsFromValidInds(validInds[ind])]
    del dbFeat_single

    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[numDb:]
    dbFeat = dbFeat[:numDb]
    print(f"dbFeat.shape = {dbFeat.shape}, qFeat.shape = {qFeat.shape}")

    qFeat_np = qFeat.detach().cpu().numpy().astype('float32')
    dbFeat_np = dbFeat.detach().cpu().numpy().astype('float32')

    db_emb, q_emb = None, None
    if opt.numSamples2Project != -1 and writer is not None:
        db_emb = TSNE(n_components=2).fit_transform(dbFeat_np[:opt.numSamples2Project])
        q_emb = TSNE(n_components=2).fit_transform(qFeat_np[:opt.numSamples2Project])

    if extract_noEval:
        return np.vstack([dbFeat_np,qFeat_np]), db_emb, q_emb, None, None

    matching_time = time.time()
    predictions, bestDists = computeMatches(opt,N_VALUES,device,dbFeat,qFeat,dbFeat_np,qFeat_np)
    
    export_predictions(predictions, eval_set)

    matching_time = time.time() - matching_time
    print(f"\n====> Query len: {qFeat_np.shape[0]}")
    print(f"====> Database len: {dbFeat_np.shape[0]}")
    print(f"====> Matching time: {matching_time:.3f} sec\n")

    # for each query get those within threshold distance
    gt,gtDists = eval_set.get_positives(retDists=True)
    gtDistsMat = cdist(eval_set.dbStruct.utmDb,eval_set.dbStruct.utmQ)

    recall_at_n = getRecallAtN(N_VALUES, predictions, gt)
    rAtL = evaluate(N_VALUES,predictions,gtDistsMat)

    recalls = {} #make dict for output
    for i,n in enumerate(N_VALUES):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if writer is not None:
            writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch)
        wandb.log({'Val/Recall@' + str(n): recall_at_n[i]},commit=False)

    return recalls, db_emb, q_emb, rAtL, predictions
