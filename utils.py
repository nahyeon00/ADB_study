import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def save_results(args, test_results):
    print('for saving results')
    
    file_name = f'{args.dataset}_{args.known_cls_ratio}_y_pred.npy'
    output_dir = os.path.join(args.output_dir, file_name)

    pred_labels_path = os.path.join(output_dir, file_name)
    np.save(pred_labels_path, test_results['y_pred'])
    true_labels_path = os.path.join(output_dir, 'y_true.npy')
    np.save(true_labels_path, test_results['y_true'])

    del test_results['y_pred']
    del test_results['y_true']

    # if not os.path.exists(args.result_dir):
    #     os.makedirs(args.result_dir)

    import datetime
    created_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    var = [args.dataset, args.known_cls_ratio, args.labeled_ratio, args.seed, created_time]
    names = ['dataset', 'known_cls_ratio', 'labeled_ratio', 'seed', 'created_time']
    vars_dict = {k:v for k,v in zip(names, var) }
    results = dict(test_results,**vars_dict)
    keys = list(results.keys())
    values = list(results.values())
    
    results_path = os.path.join(output_dir, args.results_file_name)
    
    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori,columns = keys)
        df1.to_csv(results_path,index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results,index=[1])
        df1 = df1.append(new,ignore_index=True)
        df1.to_csv(results_path,index=False)
    data_diagram = pd.read_csv(results_path)
    
    print('test_results', data_diagram)

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def F_measure(cm):
    idx = 0
    rs, ps, fs = [], [], []
    n_class = cm.shape[0]
    
    for idx in range(n_class):
        TP = cm[idx][idx]
        r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0
        rs.append(r * 100)
        ps.append(p * 100)
        fs.append(f * 100)
        
    f = np.mean(fs).round(4)
    f_seen = np.mean(fs[:-1]).round(4)
    f_unseen = round(fs[-1], 4)
    
    results = {}
    results['F1-known'] = f_seen
    results['F1-open'] = f_unseen
    results['F1'] = f
    
    return results
