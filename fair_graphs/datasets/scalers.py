"""Contains the classes definition of data scalers."""

# ----- Standard Imports

# ----- Third Party Imports
import numpy as np
import torch as tr

# ----- Library Imports


class _BaseScaler:
    def fit(self, x, update=False):
        pass
    
    def transform(self, x):
        pass
    
    def fit_transform(self, x, update=False):
        pass
    
    @staticmethod
    def validate_input(x):
        if isinstance(x, np.ndarray):
            return tr.tensor(x)
        return x

    
class MinMaxScaler(_BaseScaler):
    def __init__(self, feature_range=(0, 1), clip=True):
        assert len(feature_range) == 2, "Argument 'feature_range' must be a pair."
        assert feature_range[0] <= feature_range[1], \
            "Minimum of feature range must be less or equal than maximum."
        assert isinstance(clip, bool)
        self.feat_range = feature_range
        self.clip = clip
    
    def fit(self, x, update=False):
        #assert (not update) or hasattr(self, 'scale')
        x = _BaseScaler.validate_input(x)
        
        data_min = tr.min(x, axis=0).values
        data_max = tr.max(x, axis=0).values
        if update and hasattr(self, 'data_min'):
            data_min = tr.min(self.data_min, data_min)
            data_max = tr.max(self.data_max, data_max)
        
        data_range = data_max - data_min
        self.scale = (self.feat_range[1] - self.feat_range[0]) / data_range
        self.min = self.feat_range[0] - data_min * self.scale
        
        self.data_min = data_min
        self.data_max = data_max
        #self.data_range = data_range
        
        return self
    
    def transform(self, x):
        if not hasattr(self, 'scale'):
            raise RuntimeError("MinMaxScaler need to be fitted first.")
        
        x = _BaseScaler.validate_input(x)
        x *= self.scale
        x += self.min
        if self.clip:
            x.clamp_(min=self.feat_range[0], max=self.feat_range[1])
        return x
    
    def fit_transform(self, x, update=False):
        self.fit(x, update=update)
        return self.transform(x)