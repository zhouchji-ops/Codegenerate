"""Pure admission checks for fixed training-only observation-pose plans."""
import math
from numbers import Real

def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(label + ' must be a finite number')
    return float(value)

def validate_pose_plan(yaws, retreat):
    """Return normalized (yaw list, unsigned retreat metres), never add angles."""
    if not isinstance(yaws, (list, tuple)) or not 1 <= len(yaws) <= 32:
        raise ValueError('Provide 1..32 fixed yaw angles')
    angles=[_finite_number(v,'yaw') for v in yaws]
    if any(abs(v)>math.pi for v in angles):
        raise ValueError('Yaw magnitude must be <= pi')
    if len(set(angles))!=len(angles):
        raise ValueError('Yaw angles must be unique')
    distance=_finite_number(retreat,'retreat')
    if not 0 <= distance <= .5:
        raise ValueError('Retreat must be in [0, 0.5] metres')
    return angles,distance

def retreat_delta(axis, toward, distance):
    """axis: positive joint world direction; toward: base-to-sink vector.
    Return signed joint displacement AWAY from sink. Near-orthogonal geometry
    is rejected even for zero distance, so a caller must explicitly skip this
    calculation when no retreat is requested.
    """
    distance=_finite_number(distance,'distance')
    if not 0 <= distance <= .5:
        raise ValueError('Distance must be in [0, 0.5]')
    if len(axis)!=3 or len(toward)!=3:
        raise ValueError('Expected two three-dimensional vectors')
    a=[_finite_number(v,'axis') for v in axis]
    b=[_finite_number(v,'toward') for v in toward]
    an=math.hypot(*a);bn=math.hypot(*b)
    if an==0 or bn==0 or not math.isfinite(an) or not math.isfinite(bn):
        raise ValueError('Degenerate direction')
    dot=sum((x/an)*(y/bn) for x,y in zip(a,b))
    if abs(dot)<=.05:
        raise ValueError('Axis nearly orthogonal to sink direction')
    return -distance if dot>0 else distance

def _contact_depths(contacts):
    """Input each contact as {'pair': [geom1, geom2], 'distance_m': signed gap}.
    Duplicate contact points reduce to maximum penetration for each geom pair.
    This helper does not infer robot identity or active contacts; caller supplies
    already-classified robot-obstacle contacts, retaining near-touch records.
    """
    result={}
    for c in contacts:
        pair=c.get('pair')
        if not isinstance(pair,(list,tuple)) or len(pair)!=2 or any(not isinstance(x,str) or not x for x in pair):
            raise ValueError('Contact pair needs two nonempty geom names')
        key=tuple(sorted(pair))
        gap=_finite_number(c.get('distance_m'),'distance_m')
        depth=max(0.,-gap)
        result[key]=max(result.get(key,0.),depth)
    return result

def contact_changes(baseline,current,threshold=.0005,deepening_tol=.00005):
    """Report every new pair and every positive penetration increase.
    Hard reject: new-pair penetration > threshold OR existing-pair depth
    increase > deepening_tol. Unchanged pre-existing penetration is reported,
    not newly rejected. Threshold equality passes; callers must freeze tolerances.
    """
    threshold=_finite_number(threshold,'threshold')
    deepening_tol=_finite_number(deepening_tol,'deepening_tol')
    if threshold<0 or deepening_tol<0:
        raise ValueError('Tolerances must be nonnegative')
    old=_contact_depths(baseline);new=_contact_depths(current)
    added=[];deepened=[];hard=[]
    for pair,depth in sorted(new.items()):
        if pair not in old:
            item=dict(pair=list(pair),penetration_m=depth,hard_reject=depth>threshold)
            added.append(item)
            if item['hard_reject']:hard.append(dict(kind='new',**item))
        elif depth>old[pair]:
            delta=depth-old[pair]
            item=dict(pair=list(pair),baseline_penetration_m=old[pair],
                      penetration_m=depth,increase_m=delta,hard_reject=delta>deepening_tol)
            deepened.append(item)
            if item['hard_reject']:hard.append(dict(kind='deepened',**item))
    return dict(new_contacts=added,deepened_contacts=deepened,
                hard_reject=bool(hard),hard_violations=hard,
                baseline_pair_count=len(old),current_pair_count=len(new),
                new_threshold_m=threshold,deepening_tolerance_m=deepening_tol)

def require_train_asset(entry):
    """Only explicit train_candidate T/bridge entries; not path-based guessing."""
    if not isinstance(entry,dict) or entry.get('family') not in ('t_grip','bridge') or entry.get('split')!='train_candidate':
        raise ValueError('Only t_grip/bridge train_candidate assets are allowed')
    return entry

def changed_indices_exact(before,after,allowed):
    """Return changed scalar indices; reject undeclared changes or bad values."""
    if len(before)!=len(after):
        raise ValueError('State lengths differ')
    if any(type(i) is not int or i<0 or i>=len(before) for i in allowed):
        raise ValueError('Allowed indices must be valid integers')
    for value in list(before)+list(after):
        _finite_number(value,'state value')
    changed=[i for i,(a,b) in enumerate(zip(before,after)) if a!=b]
    if not set(changed)<=set(allowed):
        raise ValueError('Undeclared state indices changed')
    return changed
