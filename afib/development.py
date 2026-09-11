"""Small nested development search. Outer inference is a separate gated phase."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
import torch
from torch.utils.data import DataLoader

from afib.data import AFibDataset, HRVTransform, load_arrays
from afib.ltaf import digest
from afib.models import AFibModel
from afib.preprocess import external_root
from afib.train import evaluate, metrics


# Freeze this shortlist before any combined-cohort outer test is evaluated.
CANDIDATES = {'reference-bce': (False, 1.), 'se-bce': (True, 1.), 'se-sqrt-bce': (True, .5)}
POLICY = dict(epochs=20,patience=5,lr=5e-4,weight_decay=1e-4,
              checkpoint='maximum inner-validation average precision; BCE tie break',
              threshold='maximum recall subject to specificity >= 0.70; F1/specificity/threshold tie breaks',
              candidate='gate passed first, then recall, F1, average precision, AUROC, specificity',
              gate=dict(recall=.5,f1=.25,specificity=.7,average_precision='above validation prevalence'))


def threshold_metrics(labels, probabilities, threshold):
    result=metrics(labels,np.asarray(probabilities))
    binary=metrics(labels,(np.asarray(probabilities)>=threshold).astype(float))
    for key in ('accuracy','precision','recall','f1','specificity','confusion_matrix'):
        result[key]=binary[key]
    result['threshold']=float(threshold)
    return result


def choose_threshold(labels, probabilities, minimum_specificity=.7):
    y=np.asarray(labels,dtype=int); p=np.asarray(probabilities,dtype=float)
    if set(y)!={0,1} or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Threshold selection requires finite probabilities and both classes')
    # Evaluate tied scores as a unit; this exactly implements probability >= threshold.
    order=np.argsort(-p,kind='stable'); sorted_p=p[order]; sorted_y=y[order]
    ends=np.r_[np.flatnonzero(np.diff(sorted_p)!=0),len(y)-1]
    tp=np.r_[0,np.cumsum(sorted_y)[ends]]; fp=np.r_[0,ends+1-np.cumsum(sorted_y)[ends]]
    thresholds=np.r_[np.nextafter(sorted_p[0],np.inf),sorted_p[ends]]
    recall=tp/y.sum(); specificity=1-fp/(len(y)-y.sum())
    f1=np.divide(2*tp,2*tp+fp+y.sum()-tp,out=np.zeros(len(tp),dtype=float),where=(2*tp+fp+y.sum()-tp)>0)
    feasible=np.flatnonzero(specificity>=minimum_specificity)
    best=max(feasible,key=lambda i:(recall[i],f1[i],specificity[i],thresholds[i]))
    return float(thresholds[best])


def development_splits(rows, seed=2026):
    groups=np.array([r['patient'] for r in rows]); y=np.array([r['label'] for r in rows])
    for fold,(development,test) in enumerate(GroupKFold(5).split(y,y,groups)):
        inner=StratifiedGroupKFold(4,shuffle=True,random_state=seed+fold)
        local_train,local_val=next(inner.split(development,y[development],groups[development]))
        train,val=development[local_train],development[local_val]
        if len(set(y[train]))!=2 or len(set(y[val]))!=2:
            raise ValueError(f'Fold {fold}: inner partition lacks a class; inspect cohort before development')
        sets=[set(groups[i]) for i in (train,val,test)]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        yield fold,train,val,test


def save_json(path,value):
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,indent=2)); temp.replace(path)


def save_checkpoint(path,value):
    temp=path.with_suffix('.partial'); torch.save(value,temp); temp.replace(path)


def passes_gate(result):
    return (result['recall']>=.5 and result['f1']>=.25 and result['specificity']>=.7
            and result['average_precision']>result['positives']/result['n'])


def candidate_run(rows,train,val,directory,name,fold,args,manifest_sha):
    se,exponent=CANDIDATES[name]
    config=dict(candidate=name,fold=fold,seed=args.seed+fold,batch_size=args.batch_size,
                manifest_sha256=manifest_sha,policy=POLICY,se=se,weight_exponent=exponent,
                training_sha256=digest(__file__),model_sha256=digest(Path(__file__).with_name('models.py')))
    if (directory/'validation.json').exists():
        if json.loads((directory/'config.json').read_text())!=config:
            raise ValueError('Completed candidate config mismatch')
        print(f'Preserving completed {directory.name}',flush=True)
        return json.loads((directory/'validation.json').read_text())
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'config.json').exists() and json.loads((directory/'config.json').read_text())!=config:
        raise ValueError('Interrupted candidate config mismatch')
    save_json(directory/'config.json',config)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    train_rows=[rows[int(i)] for i in train]; val_rows=[rows[int(i)] for i in val]
    transform=HRVTransform().fit(np.stack([load_arrays(r)[1] for r in train_rows]))
    generator=torch.Generator().manual_seed(config['seed'])
    train_loader=DataLoader(AFibDataset(train_rows,transform),batch_size=args.batch_size,shuffle=True,generator=generator)
    val_loader=DataLoader(AFibDataset(val_rows,transform),batch_size=args.batch_size)
    model=AFibModel(se=se)
    optimizer=torch.optim.AdamW(model.parameters(),lr=POLICY['lr'],weight_decay=POLICY['weight_decay'])
    positives=sum(r['label'] for r in train_rows)
    weight=((len(train_rows)-positives)/positives)**exponent
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    history=[]; best=(-float('inf'),-float('inf')); stale=0; start_epoch=1
    if (directory/'last.pt').exists():
        state=torch.load(directory/'last.pt',weights_only=True,map_location='cpu')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']); generator.set_state(state['loader_rng'])
        history=state['history']; best=tuple(state['best']); stale=state['stale']; start_epoch=state['epoch']+1
    for epoch in range(start_epoch,POLICY['epochs']+1):
        if stale>=POLICY['patience']: break
        started=time.time(); model.train(); total=count=0
        for ecg,hrv,target in train_loader:
            optimizer.zero_grad(set_to_none=True); loss=loss_fn(model(ecg,hrv),target)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step(); total+=float(loss.detach())*len(target); count+=len(target)
        val_loss,y,p=evaluate(model,val_loader,torch.device('cpu'))
        scores=metrics(y,p); score=(scores['average_precision'],-val_loss)
        history.append(dict(epoch=epoch,train_loss=total/count,validation_loss=val_loss,
                            validation=scores,last_gradient_norm=float(norm),seconds=time.time()-started))
        if score>best:
            best=score; stale=0
            save_checkpoint(directory/'best.pt',dict(model=model.state_dict(),transform=transform.state(),
                config=config,epoch=epoch,validation_loss=val_loss,selection=POLICY['checkpoint'],
                parameters=sum(p.numel() for p in model.parameters())))
        else: stale+=1
        save_checkpoint(directory/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),loader_rng=generator.get_state(),history=history,best=best,
            stale=stale,epoch=epoch))
        save_json(directory/'history.json',history)
        print(json.dumps(dict(candidate=name,fold=fold,**history[-1])),flush=True)
    checkpoint=torch.load(directory/'best.pt',weights_only=True,map_location='cpu'); model.load_state_dict(checkpoint['model'])
    _,y,p=evaluate(model,val_loader,torch.device('cpu'))
    threshold=choose_threshold(y,p); selected=threshold_metrics(y,p,threshold)
    result=dict(candidate=name,selected_epoch=checkpoint['epoch'],epochs_run=len(history),
                threshold=threshold,validation=selected,gate_passed=passes_gate(selected),
                checkpoint_sha256=digest(directory/'best.pt'))
    predictions=[dict(patient=r['patient'],start_sample=r['start_sample'],label=int(label),probability=float(prob))
                 for r,label,prob in zip(val_rows,y,p)]
    save_json(directory/'validation_predictions.json',predictions)
    save_json(directory/'validation.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True); parser.add_argument('--run',required=True)
    parser.add_argument('--phase',choices=['develop','evaluate'],default='develop')
    parser.add_argument('--folds',nargs='+',type=int,default=list(range(5)))
    parser.add_argument('--batch-size',type=int,default=8); parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args(); root=external_root(args.root)
    if Path(args.run).name!=args.run or args.batch_size<1 or any(f not in range(5) for f in args.folds):
        raise ValueError('Invalid arguments')
    rows=json.loads((root/'manifest.json').read_text()); sha=digest(root/'manifest.json')
    out=root/'runs'/args.run; out.mkdir(parents=True,exist_ok=True)
    plan=dict(manifest_sha256=sha,policy=POLICY,candidates=CANDIDATES,seed=args.seed,batch_size=args.batch_size,
              training_sha256=digest(__file__),model_sha256=digest(Path(__file__).with_name('models.py')))
    plan=json.loads(json.dumps(plan))
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=plan:
        raise ValueError('Frozen plan mismatch')
    if not (out/'plan.json').exists():
        save_json(out/'plan.json',plan)
        save_json(out/'provenance.json',dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                                           torch_version=str(torch.__version__),device='cpu'))
    splits=list(development_splits(rows,args.seed))
    if args.phase=='evaluate':
        # All five inner selections must be completed and pass before any outer inference.
        for fold,_,_,_ in splits:
            path=out/f'fold{fold}/selection.json'
            if not path.exists() or not json.loads(path.read_text())['gate_passed']:
                raise ValueError('Validation gate not passed in all five folds; outer tests remain untouched')
    for fold,train,val,test in splits:
        if fold not in args.folds: continue
        directory=out/f'fold{fold}'; directory.mkdir(exist_ok=True)
        split={name:dict(indices=indices.tolist(),patients=sorted({rows[int(i)]['patient'] for i in indices}),
                        labels=dict(Counter(rows[int(i)]['label'] for i in indices)))
               for name,indices in zip(('train','validation','test'),(train,val,test))}
        split=json.loads(json.dumps(split))
        if (directory/'split.json').exists() and json.loads((directory/'split.json').read_text())!=split:
            raise ValueError('Frozen split mismatch')
        if not (directory/'split.json').exists(): save_json(directory/'split.json',split)
        if args.phase=='develop':
            if (directory/'selection.json').exists():
                print(f'Preserving completed inner fold {fold}',flush=True); continue
            results=[candidate_run(rows,train,val,directory/name,name,fold,args,sha) for name in CANDIDATES]
            def rank(r):
                m=r['validation']; return (r['gate_passed'],)+tuple(m[k] for k in ('recall','f1','average_precision','auroc','specificity'))
            selected=max(results,key=rank)
            save_json(directory/'selection.json',selected)
            print(json.dumps(dict(fold=fold,selection=selected)),flush=True)
        else:
            if (directory/'test_metrics.json').exists():
                print(f'Preserving completed outer fold {fold}',flush=True); continue
            selected=json.loads((directory/'selection.json').read_text()); path=directory/selected['candidate']/'best.pt'
            if digest(path)!=selected['checkpoint_sha256']: raise ValueError('Selected weights changed')
            ck=torch.load(path,weights_only=True,map_location='cpu')
            model=AFibModel(se=ck['config']['se']); model.load_state_dict(ck['model'])
            test_rows=[rows[int(i)] for i in test]
            loader=DataLoader(AFibDataset(test_rows,HRVTransform.from_state(ck['transform'])),batch_size=args.batch_size)
            loss,y,p=evaluate(model,loader,torch.device('cpu'))
            predictions=[dict(patient=r['patient'],start_sample=r['start_sample'],label=int(a),probability=float(b))
                         for r,a,b in zip(test_rows,y,p)]
            save_json(directory/'test_predictions.json',predictions)
            result=dict(fold=fold,candidate=selected['candidate'],selected_epoch=ck['epoch'],test_loss=loss,
                        test=threshold_metrics(y,p,selected['threshold']))
            save_json(directory/'test_metrics.json',result); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
