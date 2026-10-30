from pathlib import Path
import argparse, hashlib, json
import cv2
import numpy as np

def check(data):
    hashes={};counts={};labels_checked=0;real_masks=0
    for task in ['task1','task2']:
        counts[task]={}
        for split in ['train','val','test']:
            root=data/task/split;inputs=sorted((root/'inputs').glob('*.json'))
            counts[task][split]=len(inputs)
            assert len(inputs)==dict(train=64,val=16,test=16)[split],(task,split,len(inputs))
            assert len(list((root/'images').glob('*.png')))==len(inputs)
            if split=='test':assert not (root/'labels').exists(),'test labels leaked into candidate pack'
            for p in inputs:
                inp=json.loads(p.read_text(encoding='utf-8'));ip=root/inp['image'];im=cv2.imread(str(ip))
                assert im is not None and im.shape==(720,960,3),str(ip)
                assert 'H_img_to_mm' not in inp and 'L_mm' not in inp and 'd_mm' not in inp
                sha=hashlib.sha256(ip.read_bytes()).hexdigest()
                assert sha not in hashes,('duplicate synthetic image',ip,hashes.get(sha));hashes[sha]=str(ip)
                if split!='test':
                    lab=json.loads((root/'labels'/p.name).read_text(encoding='utf-8'))
                    m=cv2.imread(str(root/'labels'/lab['mask']),0)
                    assert m is not None and m.shape==im.shape[:2] and set(np.unique(m)).issubset({0,255})
                    assert np.count_nonzero(m)>30
                    labels_checked+=1
    real=data/'real_project';frames=sorted((real/'frames').glob('*.jpg'))
    assert len(frames)==24
    for p in frames:
        im=cv2.imread(str(p));assert im is not None
        ann=json.loads((real/'annotations'/f'{p.stem}.json').read_text(encoding='utf-8'))
        assert im.shape[:2]==(ann['height'],ann['width']) and ann['metric_truth_available'] is False
        for obj in ann['instances']:
            x0,y0,x1,y1=obj['crop_xyxy'];m=cv2.imread(str(real/obj['mask']),0)
            assert 0<=x0<x1<=im.shape[1] and 0<=y0<y1<=im.shape[0]
            assert m is not None and m.shape==(y1-y0,x1-x0) and set(np.unique(m)).issubset({0,255})
            real_masks+=1
    crops=sorted((real/'bolt_crops/images').glob('*.png'));assert len(crops)==48
    for p in crops:
        im=cv2.imread(str(p));m=cv2.imread(str(real/'bolt_crops/masks'/p.name),0)
        assert im is not None and m is not None and m.shape==im.shape[:2] and set(np.unique(m)).issubset({0,255})
    return dict(status='passed',synthetic_counts=counts,unique_synthetic_hashes=len(hashes),public_synthetic_labels_checked=labels_checked,real_frames=len(frames),real_frame_masks_checked=real_masks,real_bolt_crops=len(crops),hidden_answers_absent=True,metric_truth_for_real_photos=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data');p.add_argument('--output');a=p.parse_args()
    result=check(Path(a.data));s=json.dumps(result,ensure_ascii=False,indent=2);print(s)
    if a.output:Path(a.output).write_text(s,encoding='utf-8')
