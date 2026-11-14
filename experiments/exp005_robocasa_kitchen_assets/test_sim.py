"""Compile the object MJCF with a free body and test physical openings and settling."""
from pathlib import Path
import sys
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj

def run(path, scale=1.):
    path=path.resolve()
    name=path.parent.name
    root=ET.parse(path).getroot()
    compiler=root.find('compiler')
    if compiler is None:compiler=ET.SubElement(root,'compiler')
    meshdir=compiler.attrib.pop('meshdir','')
    for mesh in root.findall('asset/mesh'):
        if mesh.get('file'):mesh.set('file',str((path.parent/meshdir/mesh.get('file')).resolve()))
    texture_dir = compiler.attrib.pop('texturedir', '')
    for texture in root.findall('asset/texture'):
        if texture.get('file'):
            texture.set('file', str((path.parent / texture_dir / texture.get('file')).resolve()))
    world=root.find('worldbody')
    body=world.find('body')
    body.set('name','test_body');ET.SubElement(body,'freejoint',name='test_free')
    plane=ET.SubElement(world,'geom',name='table',type='plane',pos='0 0 -0.02',size='1 1 .1',group='4')
    m=mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    d=mj.MjData(m);mj.mj_forward(m,d)
    mask=np.array([1,0,0,0,0,0],dtype=np.uint8)
    def ray(origin,direction):
        idx=np.array([-1],dtype=np.int32)
        distance=mj.mj_ray(m,d,np.array(origin,float)*scale,np.array(direction,float),mask,1,-1,idx)
        return dict(hit=m.geom(int(idx[0])).name if idx[0]>=0 else None,d=float(distance))
    tests={}
    if name in ('scoop','mug'):
        tests['cavity']=ray([0,0,.18],[0,0,-1])
        tests['handle_hole']=ray([.049 if name=='scoop' else .052,-.1,.043 if name=='scoop' else .051],[0,1,0])
        tests['handle_solid']=ray([.053 if name=='scoop' else .069,-.1,.043 if name=='scoop' else .052],[0,1,0])
        checks=dict(open_cavity_with_floor=.155*scale<tests['cavity']['d']<.18*scale,open_handle=tests['handle_hole']['hit'] is None,
                    solid_handle=tests['handle_solid']['hit'] is not None)
    else:
        x=-.12
        for i in range(6):
            f=(i+.5)/6
            xa,xb=-.178+.003*f,-.072-.004*abs(2*f-1)
            t=(x-xa)/(xb-xa)
            y=(-.029+.06*f)*(1-t)+(-.0092+.0184*f)*t
            tests['slot'+str(i)]=ray([x,y,.05],[0,0,-1])
            if i:
                tests['rib'+str(i)]=ray([x,(y+previous)/2,.05],[0,0,-1])
            previous=y
        tests['grip_eye']=ray([.10175,0,.05],[0,0,-1])
        checks={k:(v['hit'] is not None if k.startswith('rib') else v['hit'] is None) for k,v in tests.items()}
    jid=m.joint('test_free').id;qa=m.jnt_qposadr[jid]
    dt=m.opt.timestep
    drops=[]
    for pitch in (0.,.13,-.13):
        mj.mj_resetData(m,d);d.qpos[qa+2]+=.035;d.qpos[qa+3:qa+7]=[np.cos(pitch/2),0,np.sin(pitch/2),0]
        history=[];peak=0
        for i in range(round(2/dt)):
            mj.mj_step(m,d)
            ds=[max(0.,-float(c.dist)) for c in d.contact if m.geom('table').id in (c.geom1,c.geom2)]
            if ds:peak=max(peak,max(ds))
            if i>=round(1.75/dt):history.append(d.qpos[qa:qa+3].copy())
        finite=np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
        moving=float(np.max(np.linalg.norm(np.asarray(history)-history[0],axis=1)))
        drops.append(dict(pitch_rad=pitch,peak_penetration_mm=1000*peak,final_contacts=d.ncon,
                          final_speed_m_s=float(np.linalg.norm(d.qvel[:3])),last_quarter_second_drift_mm=1000*moving,
                          stable=bool(finite and d.ncon>0 and moving<.001 and np.linalg.norm(d.qvel[:3])<.01)))
    return dict(asset=name,mujoco=mj.__version__,mass_kg=float(m.body_mass.sum()),geom_count=m.ngeom-1,
                collision_rays=tests,ray_checks=checks,drops=drops,all_pass=all(checks.values()) and all(a['stable'] for a in drops))

if __name__=='__main__':
    root=Path(sys.argv[1])
    calibration = root.parent / "calibration.json"
    scales = json.loads(calibration.read_text())["scales"] if calibration.exists() else {}
    reports=[run(root/name/'model.xml', scales.get(name, 1.)) for name in ('mug','scoop','turner')]
    (root/'validation.json').write_text(json.dumps(reports,indent=2))
    for r in reports:print(r['asset'],r['all_pass'],r['ray_checks'],[d['stable'] for d in r['drops']])
