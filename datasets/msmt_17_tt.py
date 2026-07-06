from .msmt17 import MSMT17


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
            self._print_query_gallery_coverage()

    def _print_query_gallery_coverage(self):
        gallery_cams_by_pid = {}
        for _, pid, camid, _ in self.gallery:
            gallery_cams_by_pid.setdefault(pid, set()).add(camid)

        valid_queries = 0
        for _, pid, camid, _ in self.query:
            gallery_cams = gallery_cams_by_pid.get(pid, ())
            if any(gallery_camid != camid for gallery_camid in gallery_cams):
                valid_queries += 1

        skipped_queries = len(self.query) - valid_queries
        print(
            "Market1501 eval coverage: valid queries {}/{}; skipped {} queries with no cross-camera gallery match".format(
                valid_queries,
                len(self.query),
                skipped_queries
            )
        )
        if skipped_queries:
            print("Warning: validation metrics are computed only over valid cross-camera queries.")
