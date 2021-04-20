from collections import OrderedDict


# credit: https://stackoverflow.com/questions/30975339/slicing-a-python-ordereddict
class SlicableOrderedDict(OrderedDict):
    def __getitem__(self, k):
        if not isinstance(k, slice):
            return OrderedDict.__getitem__(self, k)
        x = SlicableOrderedDict()
        for idx, key in enumerate(self.keys()):
            if k.start <= idx < k.stop:
                x[key] = self[key]
        return x
