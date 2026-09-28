import numpy as np
import torch
from matplotlib import pyplot as plt
from core import evaluation
import torch.nn.functional as F
import pandas as pd
import os
from sklearn.metrics import f1_score
from collections import defaultdict
from torchvision.utils import save_image


def test(net, criterion, testloader, outloader, epoch=None, **options):
    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()
    _pred_k, _pred_u, _labels = [], [], []  #list


    correct_samples = []
    wrong_samples = []

    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()    

            with torch.set_grad_enabled(False):
                x, _ = net(data, return_feature=True)
                logits, _ = criterion(x, labels, train=False)   # [batch_size, num_classes]
                
                predictions = logits.data.max(1)[1]  
                total += labels.size(0)
                correct += (predictions == labels.data).sum()

                _pred_k.append(logits.data.cpu().numpy())  # save known logits
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(outloader):
        #for data, labels in outloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):
                x, _ = net(data, return_feature=True)
                logits, _ = criterion(x, labels, train=False)
                _pred_u.append(logits.data.cpu().numpy())  # save unknown logits

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

   
    # concatenate logits
    _pred_k = np.concatenate(_pred_k, 0)  ## [num_known_samples, num_classes]
    _pred_u = np.concatenate(_pred_u, 0)  # [num_unknown_samples, num_classes]
    
    # Macro F1
    _labels = np.concatenate(_labels, 0)  #(647,)
    all_pred = np.argmax(_pred_k, axis=1)
    macro_f1 = f1_score(_labels, all_pred, average='macro', zero_division=0) * 100.0
    print('Macro F1 (%): {:.5f}'.format(macro_f1))

    # AUROC
    x1 = np.max(_pred_k, axis=1)  # known samples' max-logit scores
    x2 = np.max(_pred_u, axis=1)  # unknown samples' max-logit scores
    results = evaluation.metric_ood(x1, x2)['Bas']

    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    results['ACC'] = acc
    results['Macro_F1'] = macro_f1
    results['OSCR'] = _oscr_socre * 100.
    

    
    save_dir = '/8T/WJZZY/cxy/PCLAE-AMPN/log_result'  
   
    os.makedirs(save_dir, exist_ok=True)
  
    csv_path = os.path.join(save_dir, f'epoch_{epoch}_test_predictions.csv')
   
    all_results = correct_samples + wrong_samples
    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False)

 
    from datetime import datetime
    log_dir = '/8T/WJZZY/cxy/PCLAE-AMPN/log_results'
    os.makedirs(log_dir, exist_ok=True)

    if not hasattr(test, "log_file"):
        number_name= options.get('num_centers')
        batch = options.get('batch_size')
        models = options.get('model')
        timestamp = datetime.now().strftime('%m-%d')
        #timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        dataset_name = options.get('dataset', 'unknown')
        seed= options.get('seed')
        num= options.get('num_centers')
        #test.log_file = os.path.join(log_dir, f"k={number_name}-{timestamp}-{dataset_name}_test.log")
        known_tag = '_'.join(map(str, options.get('known', [])))
        unknown_tag = '_'.join(map(str, options.get('unknown', [])))
        test.log_file = os.path.join(
            log_dir,
            f"{models}-{batch}-{dataset_name}-{timestamp}-seed{seed}_test.log"
        )
        with open(test.log_file, 'a') as f:
            f.write(f"# dataset={dataset_name} seed={seed} known={options.get('known')} unknown={options.get('unknown')} batch={batch} model={models} num_centers={num}\n")

    with open(test.log_file, 'a') as f:
        f.write(f"Epoch {epoch}  "
                f"Acc (%): {results['ACC']:.3f}  "
                f"AUROC (%): {results['AUROC']:.3f}  "
                f"OSCR (%): {results['OSCR']:.3f}\n")
                #f"Macro F1 (%): {results['Macro_F1']:.3f}\n")


    return results
