#!/usr/bin/env python3
"""Two-fold, held-out training-calendar-week comparison with cluster bootstrap.

Seeds only change hash tie-breaking, not independent datasets. The confidence
interval resamples the 60 calendar Mondays, averaging matched seeds first.
Training-only exploratory screening; do not select a final policy here.
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def get_rows(root):
    records=[]
    files=[]
    for link,contexts in [('original',['none']),('inverse',['none','cohort','roster'])]:
        for context in contexts:
            for fold in (0,1):
                path=root/(link+'_'+context)/('fold_'+str(fold))/'heldout_scores.json'
                if not path.exists():
                    continue
                src=json.loads(path.read_text())
                files.append(str(path))
                for row in src.get('per_calendar_week',[]):
                    records.append(dict(row,link=link,context=context,fold=fold))
    return records,files


def mean_with_clustering(records,group_key):
    """Case-weighted mean over sites/seeds, return 60 calendar-week totals."""
    by=defaultdict(lambda:[0.,0.])
    for row in records:
        k=(row['calendar_week'],int(row['seed']))
        by[k][0]+=float(row['regret'])
        by[k][1]+=float(row['cases'])
    if not by:
        return None
    agg=defaultdict(list)
    for (monday,seed),(regret,cases) in by.items():
        agg[monday].append((regret,cases))
    return {day:(sum(v[0]/v[1] for v in vals)/len(vals)) for day,vals in agg.items()}


def cluster_bootstrap(pairs,nboot=1500,seed=718):
    """Bootstrap case-weighted paired differences across calendar Mondays."""
    groups=defaultdict(list)
    for item in pairs:
        groups[item['calendar_week']].append(item)
    if not groups:
        return None
    week_list=sorted(groups)
    # Average across matched hash seeds within each calendar week (equal cases).
    by_week=[]
    for day in week_list:
        items=groups[day]
        all_seeds={int(x['seed']) for x in items}
        seed_records=[]
        for s in all_seeds:
            subset=[x for x in items if int(x['seed'])==s]
            num=sum(x['difference'] for x in subset)
            den=sum(x['cases'] for x in subset)
            seed_records.append((num,den))
        if len(seed_records)!=len(all_seeds):
            raise ValueError('Incomplete paired seed')
        by_week.append((np.mean([r[0] for r in seed_records]),
                        np.mean([r[1] for r in seed_records])))
    a=np.array(by_week)
    estimate=float(a[:,0].sum()/a[:,1].sum())
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(week_list),size=(nboot,len(week_list)))
    stats=a[draws].sum(axis=1)
    ratio=stats[:,0]/stats[:,1]
    return dict(estimate=estimate,ci_low=float(np.quantile(ratio,.025)),
                ci_high=float(np.quantile(ratio,.975)),
                n_calendar_weeks=len(week_list),n_seed_variants=len({
                    int(x['seed']) for x in pairs}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    records,files=get_rows(args.root)
    rows=defaultdict(list)
    paired=defaultdict(list)
    indexed={(x['link'],x['context'],x['fold'],x['seed'],x['calendar_week'],
              x['site'],x['method']):x for x in records}
    for row in records:
        k=(row['link'],row['context'],row['method'],row['fold'],row['seed'])
        rows[k].append(row)
        if row['method']=='CASE_BIC':
            continue
        baseline_key=(row['link'],row['context'],row['fold'],row['seed'],
                      row['calendar_week'],row['site'],'CASE_BIC')
        if baseline_key not in indexed:
            continue
        b=indexed[baseline_key]
        if int(b['cases'])!=int(row['cases']):
            raise RuntimeError('Paired comparison case counts differ')
        paired[(row['link'],row['context'],row['method'])].append(dict(
            calendar_week=row['calendar_week'],seed=row['seed'],
            site=row['site'],cases=row['cases'],
            difference=float(row['regret'])-float(b['regret'])))
    summary=[]
    for (link,context,method,fold,seed),items in sorted(rows.items()):
        summary.append(dict(link=link,context=context,method=method,
                            eval_fold=fold,seed=seed,
                            cases=sum(int(x['cases']) for x in items),
                            weeks=len({x['calendar_week'] for x in items}),
                            regret_per_case=sum(float(x['regret']) for x in items)/
                                        sum(int(x['cases']) for x in items)))
    differences=[]
    for (link,context,method),items in sorted(paired.items()):
        stats=cluster_bootstrap(items)
        differences.append(dict(link=link,context=context,method=method,
                                versus='CASE_BIC',**stats))
    # Compare weekly context to case-only predictor for the SAME method/seed/week.
    ctxdiff=defaultdict(list)
    for row in records:
        if row['link']!='inverse' or row['context']=='none':
            continue
        control=indexed.get(('inverse','none',row['fold'],row['seed'],
                             row['calendar_week'],row['site'],row['method']))
        if control is not None:
            if int(control['cases'])!=int(row['cases']):
                raise RuntimeError('Context comparison has unequal cohort')
            ctxdiff[(row['context'],row['method'])].append(dict(
                calendar_week=row['calendar_week'],seed=row['seed'],
                site=row['site'],cases=row['cases'],
                difference=float(row['regret'])-float(control['regret'])))
    context_differences=[]
    for (context,method),items in sorted(ctxdiff.items()):
        context_differences.append(dict(context=context,method=method,
                                        versus='none',**cluster_bootstrap(items)))
    out=dict(protocol='held-out odd/even TRAIN calendar Mondays; evaluation outcomes never used in fitting',
             seed_interpretation='not independent, same weeks with alternative deterministic ties',
             cases='case-weighted',source_files=files,
             per_fold_seed=summary,
             paired_to_case=differences,
             context_minus_none=context_differences)
    output=args.root/'fold_heldout_summary.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,indent=2)+'\n')
    print('Saved',output)
    for row in differences:
        print(row['link'],row['context'],row['method'],
              'minus CASE_BIC',round(row['estimate'],4),
              'CI',round(row['ci_low'],4),round(row['ci_high'],4))
    if not files:
        print('No held-out scores found yet. Run score stage after CASE fit.')


if __name__=='__main__':
    main()
