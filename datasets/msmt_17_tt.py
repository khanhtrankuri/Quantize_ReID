from .msmt17 import MSMT17
from .reid_audit import audit_reid_split


class MSMT17_datatt(MSMT17):
    dataset_dir = 'MSMT17_datatt'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(MSMT17_datatt, self).__init__(
            root=root,
            verbose=False,
            pid_begin=pid_begin,
            **kwargs
        )
        if verbose:
            print("=> MSMT17_datatt loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)
            audit_reid_split(self.raw_train, self.raw_query, self.raw_gallery)
