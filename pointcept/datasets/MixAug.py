from .builder import build_dataset
from .transform import TRANSFORMS

@TRANSFORMS.register_module()
class MixData(object):
    def __init__(self,  dataset, **kwargs):
        super(MixData, self).__init__()
        self.dataset = build_dataset(dataset)
        pass

    def __call__(self, data_dict):
        data = dict()

        return data