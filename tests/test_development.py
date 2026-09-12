import numpy as np
import pytest
import torch
from types import SimpleNamespace

from afib.development import choose_threshold, development_splits, threshold_metrics, passes_gate


def test_threshold_matches_exhaustive_validation_search_with_ties():
    y=np.array([0,1,0,1,0,0,0,1,0,0])
    p=np.array([.1,.8,.8,.8,.3,.3,.2,.4,.4,.2])
    thresholds=np.r_[np.nextafter(p.max(),np.inf),np.unique(p)]
    results=[threshold_metrics(y,p,t) for t in thresholds]
    expected=max((r for r in results if r['specificity']>=.7),
                 key=lambda r:(r['recall'],r['f1'],r['specificity'],r['threshold']))
    assert choose_threshold(y,p)==expected['threshold']
    actual=threshold_metrics(y,p,choose_threshold(y,p))
    assert actual==expected
    assert actual['auroc']==threshold_metrics(y,p,.99)['auroc']


def test_threshold_cannot_claim_sensitivity_from_constant_scores():
    y=np.array([0,0,0,1]); p=np.ones(4)*.4
    result=threshold_metrics(y,p,choose_threshold(y,p))
    assert result['recall']==0 and result['specificity']==1
    assert not passes_gate(result)
    with pytest.raises(ValueError): choose_threshold(np.zeros(4),p)
    with pytest.raises(ValueError): choose_threshold(y,np.full(4,np.nan))


def test_nested_groups_and_outer_once_coverage():
    rows=[dict(patient=f'dataset:{p}',label=int(i<3)) for p in range(40) for i in range(10+p%3)]
    all_test=[]
    for fold,train,val,test in development_splits(rows):
        parts=[{rows[int(i)]['patient'] for i in indices} for indices in (train,val,test)]
        assert not (parts[0]&parts[1] or parts[0]&parts[2] or parts[1]&parts[2])
        assert {rows[int(i)]['label'] for i in val}=={0,1}
        all_test.extend(test)
    assert sorted(all_test)==list(range(len(rows)))


def test_training_resume_preserves_completed_epochs_and_ignores_outer_rows(monkeypatch,tmp_path):
    from afib import development as dev
    class TinyModel(torch.nn.Module):
        def __init__(self,se=False):
            super().__init__(); self.head=torch.nn.Linear(6,1)
        def forward(self,ecg,hrv): return self.head(hrv.mean(dim=1)).squeeze(-1)
    monkeypatch.setattr(dev,'AFibModel',TinyModel)
    monkeypatch.setitem(dev.POLICY,'epochs',2)
    rows=[]
    for i in range(6):
        path=tmp_path/f'{i}.npz'
        np.savez_compressed(path,ecg=np.zeros((20,1,7500),dtype=np.float32),
                            hrv=np.full((20,6),float(i%2),dtype=np.float32))
        rows.append(dict(patient=str(i),start_sample=0,label=i%2,path=str(path)))
    # Any attempt to load outer-test signal would fail.
    rows.append(dict(patient='outer',start_sample=0,label=1,path='MUST_NOT_BE_READ.npz'))
    args=SimpleNamespace(seed=2026,batch_size=2,threads=1)
    directory=tmp_path/'candidate'
    result=dev.candidate_run(rows,[0,1,2,3],[4,5],directory,'reference-bce',0,args,'manifest-test')
    original=(directory/'best.pt').read_bytes()
    assert result['epochs_run']==2
    assert dev.candidate_run(rows,[0,1,2,3],[4,5],directory,'reference-bce',0,args,'manifest-test')==result
    (directory/'validation.json').unlink()  # Simulate interruption before completion marker.
    resumed=dev.candidate_run(rows,[0,1,2,3],[4,5],directory,'reference-bce',0,args,'manifest-test')
    assert resumed['epochs_run']==2 and resumed['selected_epoch']==result['selected_epoch']
    assert (directory/'best.pt').read_bytes()==original
    assert not (directory/'test_metrics.json').exists()


def test_development_rejects_missing_or_changed_cohort_audit(tmp_path):
    import json
    from afib.development import load_audited_cohort
    from afib.ltaf import digest
    (tmp_path/'manifest.json').write_text('[]')
    with pytest.raises(FileNotFoundError): load_audited_cohort(tmp_path)
    audit=dict(manifest_sha256=digest(tmp_path/'manifest.json'),label_policy='onset-v2',
               record_groups_screened=108,split_audit='checked')
    (tmp_path/'cohort-audit.json').write_text(json.dumps(audit))
    with pytest.raises(ValueError,match='Complete combined'): load_audited_cohort(tmp_path)
    audit['record_groups_screened']=109
    (tmp_path/'cohort-audit.json').write_text(json.dumps(audit))
    (tmp_path/'splits.json').write_text(json.dumps([{}]*5))
    assert load_audited_cohort(tmp_path)[0]==[]
    (tmp_path/'manifest.json').write_text('[{}]')
    with pytest.raises(ValueError,match='Complete combined'): load_audited_cohort(tmp_path)
