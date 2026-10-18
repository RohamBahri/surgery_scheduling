#!/usr/bin/env python3
"""Summarize training comparisons; never reads validation/test files."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    observations=[]; errors=[]
    for link in ('original','inverse'):
        for context in ('none','weekly'):
            folder=args.root/(link+'_'+context)
            file=folder/'pilot_training_comparison.json'
            if not file.exists():
                errors.append('Missing '+str(file))
                continue
            report=json.loads(file.read_text())
            for seed,methods in report['seeds'].items():
                for method,result in methods.items():
                    if result.get('complete') and 'regret_per_case' in result:
                        observations.append((link,context,int(seed),method,
                                             float(result['regret_per_case'])))
                    else:
                        errors.append(str((link,context,seed,method,result.get('error'))))
    groups=defaultdict(list);lookup={}
    for link,ctx,seed,method,value in observations:
        groups[(link,ctx,method)].append(value)
        lookup[(link,ctx,seed,method)]=value
    rows=[]
    for (link,ctx,method),values in sorted(groups.items()):
        rows.append(dict(link=link,context=ctx,method=method,
                         n_seeds=len(values),mean_regret_per_case=sum(values)/len(values),
                         min_regret_per_case=min(values),max_regret_per_case=max(values)))
    paired=[]
    for link in ('original','inverse'):
        for ctx in ('none','weekly'):
            for method in ('VF','VF_GAP','ANCHORED'):
                diffs=[val-lookup[(link,ctx,seed,'CASE_BIC')]
                       for (l,c,seed,m),val in lookup.items()
                       if (l,c,m)==(link,ctx,method)
                       and (link,ctx,seed,'CASE_BIC') in lookup]
                if diffs:
                    paired.append(dict(link=link,context=ctx,method=method,
                                       baseline='CASE_BIC',n_seeds=len(diffs),
                                       mean_diff=sum(diffs)/len(diffs),
                                       min_diff=min(diffs),max_diff=max(diffs)))
    summary=dict(training_only=True,rows=rows,paired=paired,errors=errors)
    args.root.mkdir(parents=True,exist_ok=True)
    outfile=args.root/'pilot_summary.json'
    outfile.write_text(json.dumps(summary,indent=2)+'\n')
    print('Saved',outfile)
    for row in rows:
        print(row['link'],row['context'],row['method'],
              row['n_seeds'],'mean',round(row['mean_regret_per_case'],4))
    if errors:
        print('Incomplete:',len(errors),'records; inspect summary errors')


if __name__=='__main__':
    main()
