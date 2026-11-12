from pathlib import Path
from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToSink
from robocasa.models.objects.kitchen_object_utils import OBJ_CATEGORIES
class RigidPilotCounterToSink(PickPlaceCounterToSink):
    def __init__(self,pilot_xml,*args,**kwargs):
        self.pilot_xml=str(Path(pilot_xml).resolve())
        super().__init__(*args,**kwargs)
    def sample_object(self,groups,**kwargs):
        if groups=='__rigid_pilot__':
            matches=[]
            for category in OBJ_CATEGORIES.values():
                for entry in category.values():
                    for path in getattr(entry,'mjcf_paths',[]):
                        if Path(path).resolve()==Path(self.pilot_xml):matches.append(path)
            if not matches:raise ValueError('native asset absent from registry after resolved-path match')
            groups=matches[0]
        return super().sample_object(groups,**kwargs)
