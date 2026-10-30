"""Offline recruitment evaluator. Scores 75 objective points; 25 require review.

All expected samples are in the denominator. Missing/invalid outputs incur
zero IoU and capped geometric-error penalties. No metric truth for real photos.
"""
from pathlib import Path
import argparse, json, math
import numpy as np
import cv2

def readj(p): return json.loads(p.read_text(encoding='utf-8'))
def predictions(root, task):
    p=root/task/'results.json'
    if not p.exists():return {}
    rows=readj(p)
    if not isinstance(rows,list):raise ValueError(f'{p}: expected a list')
    out={}
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get('sample_id'),str):
            raise ValueError(f'{p}: each row needs sample_id')
        if row['sample_id'] in out:raise ValueError('duplicate sample_id '+row['sample_id'])
        out[row['sample_id']]=row
    return out
def mask_iou(row,root,truth):
    if row.get('status')!='ok':return 0.,'status_not_ok'
    name=row.get('mask')
    if not isinstance(name,str):return 0.,'mask_missing'
    p=(root/name).resolve()
    if not p.is_relative_to(root.resolve()):return 0.,'mask_path_outside_prediction_folder'
    m=cv2.imread(str(p),0) if p.exists() else None
    if m is None or m.shape!=truth.shape:return 0.,'mask_missing_or_wrong_size'
    if not set(np.unique(m)).issubset({0,255}):return 0.,'mask_must_be_binary_0_255'
    a=m>0;b=truth>0;union=np.count_nonzero(a|b)
    return float(np.count_nonzero(a&b)/union) if union else 1.,'ok'
def finite_number(v):
    return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def stats(values):
    if not values:return dict(count=0,MAE_mm=None,RMSE_mm=None,P95_mm=None,max_mm=None)
    v=np.array(values,float)
    return dict(count=len(v),MAE_mm=float(v.mean()),RMSE_mm=float(np.sqrt(np.mean(v*v))),P95_mm=float(np.percentile(v,95)),max_mm=float(v.max()))
def main(args):
    data=Path(args.data);pred=Path(args.pred);truth=Path(args.truth) if args.truth else data
    report=dict(split=args.split,objective_max=75,human_review_max=25,truth_scope='synthetic_geometry_only',tasks={})
    for task in ['task1','task2']:
        expected=sorted((data/task/args.split/'inputs').glob('*.json'))
        if not expected:raise ValueError(f'no inputs for {task} {args.split}')
        labels=truth/task/args.split/'labels'
        if not labels.is_dir():raise ValueError('labels not found; test truth is in the organizer-only package')
        rows=predictions(pred,task);details=[];ious=[];errs=[];Ls=[];ds=[];valid=0;raw_points=[];raw_L=[];raw_d=[]
        missing=0
        ids={p.stem for p in expected}
        for ip in expected:
            sid=ip.stem;inp=readj(ip);lab=readj(labels/f'{sid}.json')
            row=rows.get(sid,{})
            if not row:missing+=1
            gt=cv2.imread(str(labels/lab['mask']),0)
            if gt is None:raise ValueError('unreadable truth mask')
            iou,reason=mask_iou(row,pred/task,gt);ious.append(iou)
            rec=dict(sample_id=sid,iou=iou,mask_status=reason)
            if task=='task1':
                e=None
                try:
                    H=np.array(row['H_img_to_mm'],dtype=float)
                    if row.get('status')!='ok' or H.shape!=(3,3) or not np.isfinite(H).all() or np.linalg.matrix_rank(H)<3:raise ValueError('invalid H')
                    p=np.array(inp['queries_px'],float);q=np.c_[p,np.ones(len(p))]@H.T
                    if np.any(np.abs(q[:,2])<1e-10):raise ValueError('point at infinity')
                    xy=q[:,:2]/q[:,2,None]
                    err=np.linalg.norm(xy-np.array(lab['queries_mm']),axis=1)
                    if not np.isfinite(err).all():raise ValueError('nonfinite mapped points')
                    if row.get('shape')!=lab['shape']:raise ValueError('wrong shape')
                    e=float(err.mean());valid+=1;raw_points.extend(err.tolist())
                except (KeyError,ValueError,TypeError,np.linalg.LinAlgError):pass
                errs.append(20. if e is None else min(e,20.))
                rec['point_mean_error_mm']=e
            else:
                a=row.get('L_mm');b=row.get('d_mm')
                ok=row.get('status')=='ok' and finite_number(a) and finite_number(b) and a>0 and b>0
                el=abs(a-lab['L_mm']) if ok else None;ed=abs(b-lab['d_mm']) if ok else None
                if ok:raw_L.append(el);raw_d.append(ed)
                Ls.append(20. if el is None else min(el,20.));ds.append(6. if ed is None else min(ed,6.))
                valid+=int(ok);rec.update(length_error_mm=el,diameter_error_mm=ed)
            details.append(rec)
        miou=float(np.mean(ious))
        out=dict(expected=len(expected),missing=missing,extra_sample_ids=sorted(set(rows)-ids),valid_geometry_fraction=valid/len(expected),mean_IoU=miou,per_sample=details)
        if task=='task1':
            score_iou=15*float(np.clip((miou-.5)/.45,0,1))
            score_calib=20*max(0,1-float(np.mean(errs))/10)
            out.update(valid_only_point_errors=stats(raw_points),point_error_capped_mean_mm=float(np.mean(errs)),segmentation_points=score_iou,calibration_points=score_calib,objective_points=score_iou+score_calib,objective_max=35)
        else:
            score_iou=10*float(np.clip((miou-.5)/.45,0,1))
            score_L=15*max(0,1-float(np.mean(Ls))/10)
            score_d=15*max(0,1-float(np.mean(ds))/3)
            out.update(valid_only_length_errors=stats(raw_L),valid_only_diameter_errors=stats(raw_d),length_error_capped_mean_mm=float(np.mean(Ls)),diameter_error_capped_mean_mm=float(np.mean(ds)),segmentation_points=score_iou,length_points=score_L,diameter_points=score_d,objective_points=score_iou+score_L+score_d,objective_max=40)
        report['tasks'][task]=out
    report['objective_total']=sum(x['objective_points'] for x in report['tasks'].values())
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='tasks'},ensure_ascii=False,indent=2))
    for k,v in report['tasks'].items():print(k,json.dumps({a:b for a,b in v.items() if a!='per_sample'},ensure_ascii=False))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data');p.add_argument('--pred',default='predictions');p.add_argument('--split',choices=['train','val','test'],default='val')
    p.add_argument('--truth',default=None);p.add_argument('--output',default='evaluation.json')
    main(p.parse_args())
