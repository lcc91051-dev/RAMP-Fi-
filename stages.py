import csv,math,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data import write_json,sha
from model import Network,PhaseNetwork,CSIClassifier,seed_all,transform,augment_phase
from training_utils import CycleIndices,augment,kernel_groups,bn_snapshot,restore_buffers
def state(path,device=None):
    if device is None: device='cuda:0' if torch.cuda.is_available() else 'cpu'
    return torch.load(path,map_location=device,weights_only=True)

def npredict(model,x,phase=False):
    model.eval();out=[]
    with torch.no_grad():
        for part in x.split(64):
            _,z=model(transform(part,'phase_augment') if phase else part)
            out.append(z.softmax(-1).mean(1).cpu().numpy())
    return np.concatenate(out)

def nll(prob,y):return float(-np.log(prob[np.arange(len(y)),y].clip(1e-12)).mean())

def history(out,rows):
    with (out/'history.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def finish(out,model_type,ref,best_epoch,started,**extra):
    record=dict(model_type=model_type,reference=ref,best_epoch=best_epoch,
      seconds=time.monotonic()-started,checkpoint_sha256=sha(out/'best_model.pth'),
      target_arrays_loaded=False,unlabeled_truth_used=False)
    record.update(extra);write_json(out/'complete.json',record)

def train_online(data,out,seed,ref=1,epochs=80):
    started=time.monotonic();seed_all(seed);data.load_single(ref)
    # Online pseudo-label classification from the weak-view probabilities.
    flat=CSIClassifier().to(data.device)
    model=Network.__new__(Network);torch.nn.Module.__init__(model)
    model.encoder=flat.encoder;model.classifier=flat.classifier
    cfg=dict(lr=.03,weight_decay=.001,labeled_batch=32,unlabeled_batch=96,threshold=.9)
    write_json(out/'config.json',dict(seed=seed,epochs=epochs,reference=ref,
                                      pseudo_weight=1.0,pseudo_label_rule='direct_weak_view',**cfg))
    optimizer=torch.optim.SGD(kernel_groups(flat,.001),lr=.03,momentum=.9)
    steps=max(math.ceil(data.nl/32),math.ceil(data.nu/96));best=-1.;best_epoch=0;rows=[]
    lhash=hashlib.sha256();uhash=hashlib.sha256();lcount=ucount=updates=0
    for epoch in range(epochs):
        flat.train();ls=CycleIndices(data.nl,data.device);us=CycleIndices(data.nu,data.device)
        traces=dict(indices=[],probabilities=[],accepted=[]);total=0.
        for step in range(steps):
            progress=(epoch*steps+step)/(epochs*steps)
            for group in optimizer.param_groups:group['lr']=.03*math.cos(math.acos(.25)*progress)
            li=ls.take(32);ui=us.take(96);xl=data.x_l[li].flatten(0,1)
            xu=data.x_u[ui].flatten(0,1)
            lhash.update(li.cpu().numpy().tobytes());uhash.update(ui.cpu().numpy().tobytes());lcount+=len(li);ucount+=len(ui);updates+=1
            yl=data.y_l[li].repeat_interleave(2);nl,nu=len(xl),len(xu)
            xw,xs=augment(xl),augment(xl,strong=True);uw,usx=augment(xu),augment(xu,strong=True)
            snap=bn_snapshot(flat);zs0=flat(torch.cat((xw,xs)));restore_buffers(snap)
            zs,zuw,zus=flat(torch.cat((xw,xs,uw,usx))).split((2*nl,nu,nu))
            zs=zs+(zs0-zs)*torch.rand_like(zs);zw,zstrong=zs.chunk(2)
            ql=zw.detach().softmax(-1);q=zuw.detach().softmax(-1)
            mask=q.max(-1).values>=.9*ql.max(-1).values.mean()
            lu=(F.cross_entropy(zus,q.argmax(-1),reduction='none')*mask).mean()
            traces['indices'].append(ui.repeat_interleave(2).cpu().numpy())
            traces['probabilities'].append(q.cpu().numpy());traces['accepted'].append(mask.cpu().numpy())
            ls_loss=.5*(F.cross_entropy(zw,yl)+F.cross_entropy(zstrong,yl))
            ramp=.5-.5*math.cos(math.pi*min(2*progress,1));loss=ls_loss+ramp*lu
            assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();total+=loss.item()
        pv=npredict(model,data.x_v);yv=data.y_v.cpu().numpy();acc=float((pv.argmax(-1)==yv).mean())
        if acc>=best:best=acc;best_epoch=epoch+1;torch.save(model.state_dict(),out/'best_model.pth')
        if traces['indices']:np.savez_compressed(out/f'training_pseudo_epoch_{epoch+1:03d}.npz',**{k:np.concatenate(v) for k,v in traces.items()})
        rows.append(dict(epoch=epoch+1,loss=total/steps,val_accuracy=acc,val_nll=nll(pv,yv)));history(out,rows)
    model.load_state_dict(state(out/'best_model.pth'))
    write_json(out/'exposure.json',dict(labeled_recordings=lcount,unlabeled_forwarded_recordings=ucount,optimizer_steps=updates,labeled_sequence_sha256=lhash.hexdigest(),unlabeled_sequence_sha256=uhash.hexdigest(),pseudo_loss_weight=1.0,trace_scope='online pseudo-label events'))
    finish(out,'network',ref,best_epoch,started,pseudo_label_rule='direct_weak_view')

def train_phase(data,out,seed,initial,ref=1,epochs=40):
    started=time.monotonic();seed_all(seed);data.load_single(ref)
    model=PhaseNetwork().to(data.device);model.initialize_from_semantic_model(state(initial/'best_model.pth'))
    write_json(out/'config.json',dict(seed=seed,epochs=epochs,reference=ref,lr=.0003,batch=128,
              initial_semantic_sha256=sha(initial/'best_model.pth'),within_weight=.1,selection='source val accuracy'))
    x=torch.cat((data.x_l,data.x_u));y=torch.cat((data.y_l,data.y_u));steps=math.ceil(len(x)/128)
    lhash=hashlib.sha256();lcount=ucount=updates=0
    optimizer=torch.optim.Adam(model.parameters(),lr=.0003,weight_decay=.0003)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps*epochs,eta_min=.00003)
    yv=data.y_v.cpu().numpy();pv=npredict(model,data.x_v,True);best=float((pv.argmax(-1)==yv).mean());best_epoch=0;rows=[]
    torch.save(model.state_dict(),out/'best_model.pth')
    for epoch in range(epochs):
        model.train();total=0.;order=torch.randperm(len(x),generator=torch.Generator().manual_seed(seed+epoch)).to(data.device)
        for step,idx in enumerate(order.split(128)):
            li=idx[idx<data.nl];lhash.update(li.cpu().numpy().tobytes());lcount+=len(li);ucount+=int((idx>=data.nl).sum());updates+=1
            original=x[idx];alt=augment_phase(original,seed*10000+100*epoch+step)
            h,z=model(torch.cat((transform(original,'phase_augment'),transform(alt,'phase_augment'))))
            labels=y[idx].repeat(2);lm=labels>=0
            sup=F.cross_entropy(z[lm].flatten(0,1),labels[lm].repeat_interleave(2)) if lm.any() else z.sum()*0
            pair=(1-F.cosine_similarity(h[:,0],h[:,1],dim=-1)).mean();loss=sup+.1*pair
            assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();scheduler.step();total+=loss.item()
        pv=npredict(model,data.x_v,True);acc=float((pv.argmax(-1)==yv).mean())
        if acc>=best:best=acc;best_epoch=epoch+1;torch.save(model.state_dict(),out/'best_model.pth')
        rows.append(dict(epoch=epoch+1,loss=total/steps,val_accuracy=acc,val_nll=nll(pv,yv)));history(out,rows)
    write_json(out/'exposure.json',dict(labeled_recordings=lcount,unlabeled_forwarded_recordings=ucount,optimizer_steps=updates,labeled_sequence_sha256=lhash.hexdigest()))
    finish(out,'phase',ref,best_epoch,started)
