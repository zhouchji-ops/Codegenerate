"""Read-only format audit. A pass is NOT physical or expert certification."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np


def audit(directory):
    root=Path(directory)
    result=json.loads((root/'result.json').read_text())
    n=result['steps']
    if type(n) is not int or not 0<n<=result['horizon']:
        raise ValueError('invalid step count')
    with h5py.File(root/'trajectory.hdf5','r') as h:
        for name,shape in [('actions',(n,12)),('state14',(n,14)),('success',(n,))]:
            if h[name].shape!=shape:raise ValueError(name+' shape mismatch')
        for name in ('actions','state14','sim_states','terminal_sim_state'):
            if not np.isfinite(h[name][:]).all():raise ValueError(name+' non-finite')
        if h['sim_states'].shape[0]!=n or h['terminal_sim_state'].shape!=h['sim_states'].shape[1:]:
            raise ValueError('sim state alignment')
        for key in ('video.robot0_agentview_left','video.robot0_agentview_right','video.robot0_eye_in_hand'):
            ds=h['observations/'+key]
            if ds.shape!=(n,256,256,3) or ds.dtype!=np.uint8:raise ValueError(key+' shape/dtype mismatch')
            # Read bounded blocks to exercise decompression without loading an entire episode.
            for index in range(0,n,64): _=ds[index:index+64]
        if bool(h.attrs['success'])!=bool(result['success']) or bool(h['success'][-1])!=bool(result['success']):
            raise ValueError('success metadata inconsistent')
        if h.attrs['observation_alignment']!='pre_action':raise ValueError('unexpected time alignment')
    xml=(root/'model.xml').read_bytes()
    return dict(run=root.name,integrity_pass=True,steps=n,official_success=result['success'],
                model_xml_sha256=hashlib.sha256(xml).hexdigest(),
                physical_review='not_checked_by_this_script',source='policy_rollout_not_expert')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directories',type=Path,nargs='+')
    args=p.parse_args();reports=[]
    for directory in args.directories:
        try:reports.append(audit(directory))
        except Exception as error:reports.append(dict(run=directory.name,integrity_pass=False,error=str(error)))
    print(json.dumps(reports,indent=2))
    raise SystemExit(0 if all(r['integrity_pass'] for r in reports) else 1)
