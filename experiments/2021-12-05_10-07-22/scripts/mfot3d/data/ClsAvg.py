import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class ClassAverage(object):
    def __init__(self, classes=['Cow'],
                 save_path = 'vfa/data/ClsAvg.json'):
        self.save_path = save_path
        self.dimension_map = {}
        for cls in classes:
            cls_ = cls.lower()
            if cls_ in self.dimension_map.keys():
                continue
            self.dimension_map[cls_]={}
            self.dimension_map[cls_]['total'] = np.zeros((3, ), dtype=np.float32)
            self.dimension_map[cls_]['count'] = 0
            self.dimension_map[cls_]['mean'] = 0
    
    def add_item(self, cls, dimension):
        cls_ = cls.lower()
        assert cls_ in self.dimension_map.keys()
        self.dimension_map[cls_]['total'] += dimension
        self.dimension_map[cls_]['count'] += 1
    
    def get_mean(self, cls):
        cls_ = cls.lower()
        assert cls_ in self.dimension_map.keys()
        return self.dimension_map[cls_]['total'] / self.dimension_map[cls_]['count']
    

    def dump_to_file(self):
        for cls in self.dimension_map.keys():
            self.dimension_map[cls]['mean'] = self.get_mean(cls)
        with open(self.save_path, 'w') as f:
            json.dump(self.dimension_map, f, cls=NumpyEncoder, indent=4)
    
    def load_from_file(self):
        with open(self.save_path, 'r') as f:
            result = json.load(f)
        for key, val in result.items():
            self.dimension_map[key]['count'] = val['count']
            self.dimension_map[key]['total'] = np.array(val['total'], dtype=np.float32)
            self.dimension_map[key]['mean'] = np.array(val['mean'], dtype=np.float32)