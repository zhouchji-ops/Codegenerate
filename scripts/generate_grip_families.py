"""Connected primitive grip families; handwritten, NOT Kimi output.

Generates external assets only. Split by construction family before collection.
All component boxes meet at faces (no intentionally overlapping volumes).
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def components(family, width, thickness):
    # Each tuple: full dimensions, centre, metres. Base supports all families.
    base = ((width,.05,.012),(0,0,.006))
    if family == 't_grip':
        return [base, ((.018,.025,.025),(0,0,.0245)), ((width,.025,thickness),(0,0,.037+thickness/2))]
    if family == 'bridge':
        gap=.032
        post=.008
        height=.030
        return [base, ((post,.024,height),(-(gap+post)/2,0,.012+height/2)),
                ((post,.024,height),((gap+post)/2,0,.012+height/2)),
                ((gap+2*post,.024,thickness),(0,0,.012+height+thickness/2))]
    if family == 'cross_grip':
        return [base, ((.018,.05,thickness),(0,0,.012+thickness/2)),
                (((width-.018)/2,.018,thickness),(-(width+.018)/4,0,.012+thickness/2)),
                (((width-.018)/2,.018,thickness),((width+.018)/4,0,.012+thickness/2))]
    if family == 'elbow_grip':
        return [base, ((.016,.032,.035),(-width/2+.008,0,.0295)),
                ((width-.016,.024,thickness),(.008,0,.047+thickness/2)),
                ((.016,.032,thickness),(-width/2+.008,0,.047+thickness/2))]
    raise ValueError('unknown family')


def build(family, width, thickness, mass=.04):
    parts=components(family,width,thickness)
    lows=[min(p[axis]-s[axis]/2 for s,p in parts) for axis in range(3)]
    highs=[max(p[axis]+s[axis]/2 for s,p in parts) for axis in range(3)]
    center=[(lo+hi)/2 for lo,hi in zip(lows,highs)]
    full=[hi-lo for lo,hi in zip(lows,highs)]
    volume=sum(s[0]*s[1]*s[2] for s,p in parts)
    root=ET.Element('mujoco',model=family)
    ET.SubElement(root,'asset')
    body=ET.SubElement(ET.SubElement(ET.SubElement(root,'worldbody'),'body'),'body',name='object')
    for i,(size,pos) in enumerate(parts):
        common=dict(type='box',size=' '.join(str(x/2) for x in size),pos=' '.join(str(x-c) for x,c in zip(pos,center)))
        ET.SubElement(body,'geom',name='collision_%d'%i,group='0',mass=str(mass*size[0]*size[1]*size[2]/volume),friction='0.95 0.3 0.1',rgba='0.8 0.8 0.8 0',**common)
        ET.SubElement(body,'geom',name='visual_%d'%i,group='1',mass='0',contype='0',conaffinity='0',rgba='.3 .6 .8 1',**common)
    ET.SubElement(body,'geom',name='reg_bbox',type='box',size=' '.join(str(x/2) for x in full),pos='0 0 0',group='1',mass='0',contype='0',conaffinity='0',rgba='0 1 0 0')
    return ET.tostring(root,encoding='unicode'),full,len(parts)


def generate(output, repo):
    output,repo=Path(output).resolve(),Path(repo).resolve()
    if output==repo or repo in output.parents: raise ValueError('external output required')
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    splits={'t_grip':'train_candidate','bridge':'train_candidate','cross_grip':'development_reserved','elbow_grip':'test_reserved'}
    for family,split in splits.items():
        for i,(width,thickness) in enumerate(itertools.product((.052,.060,.068),(.008,.012))):
            name=family+'_%02d'%i
            xml,size,n=build(family,width,thickness)
            folder=output/name; folder.mkdir(); data=xml.encode('utf-8'); (folder/'model.xml').write_bytes(data)
            records.append(dict(id=name,family=family,split=split,path=name+'/model.xml',full_size_m=size,
                xml_sha256=hashlib.sha256(data).hexdigest(),mass_kg=.04,components=n,
                status='generated_not_physics_or_task_validated',source='handwritten_compound_primitives'))
    (output/'manifest.json').write_text(json.dumps(dict(schema_version=1,task='PickPlaceCounterToSink',
        note='Reserved families must not guide training selection; shared appearance, nominal mass fixed, inertia varies.',candidates=records),indent=2))
    print('Generated',len(records),'candidates; 12 train, 6 development-reserved, 6 test-reserved')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();generate(args.output,Path(__file__).resolve().parents[1])
