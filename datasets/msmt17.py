
import os.path as osp

from tqdm import tqdm

from .bases import BaseImageDataset
from .reid_audit import audit_reid_split


class MSMT17(BaseImageDataset):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    dataset_dir = 'MSMT17_datatt'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(MSMT17, self).__init__()
        self.pid_begin = pid_begin
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'test')
        self.list_train_path = osp.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')

        self._check_before_run()
        train_records = self._read_list(self.list_train_path, desc="Reading list_train.txt")
        val_records = self._read_list(self.list_val_path, desc="Reading list_val.txt")
        query_records = self._read_list(self.list_query_path, desc="Reading list_query.txt")
        gallery_records = self._read_list(self.list_gallery_path, desc="Reading list_gallery.txt")
        train_pid2label = self._build_pid2label(train_records + val_records)
        self.raw_train = self._records_for_audit(train_records + val_records)
        self.raw_query = self._records_for_audit(query_records)
        self.raw_gallery = self._records_for_audit(gallery_records)

        train = self._process_dir(self.train_dir, train_records, pid2label=train_pid2label, desc="Loading train")
        val = self._process_dir(self.train_dir, val_records, pid2label=train_pid2label, desc="Loading val")
        train += val
        query = self._process_dir(
            self.test_dir,
            query_records,
            desc="Loading query",
        )
        gallery = self._process_dir(
            self.test_dir,
            gallery_records,
            desc="Loading gallery",
        )
        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)
            audit_reid_split(self.raw_train, self.raw_query, self.raw_gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))
        for list_path in [self.list_train_path, self.list_val_path, self.list_query_path, self.list_gallery_path]:
            if not osp.exists(list_path):
                raise RuntimeError("'{}' is not available".format(list_path))

    def _read_list(self, list_path, desc=None):
        records = []
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        for line in tqdm(lines, desc=desc or "Reading list", unit="line"):
            if not line.strip():
                continue
            img_path, pid = line.split()
            pid = int(pid)
            if pid < 0:
                continue
            records.append((img_path, pid))
        return records

    @staticmethod
    def _build_pid2label(records):
        pids = sorted({pid for _, pid in records})
        return {pid: label for label, pid in enumerate(pids)}

    def _records_for_audit(self, records):
        dataset = []
        for img_path, pid in records:
            camid = int(img_path.split('_')[2]) - 1
            dataset.append((img_path, pid, camid, 0))
        return dataset

    def _process_dir(self, dir_path, records, pid2label=None, desc=None):
        dataset = []
        cam_container = set()

        for img_path, pid in tqdm(records, desc=desc or "Loading dataset", unit="img"):
            label = pid2label[pid] if pid2label is not None else pid

            camid = int(img_path.split('_')[2])
            img_path = osp.join(dir_path, img_path)

            dataset.append((img_path, self.pid_begin + label, camid - 1, 0))
            cam_container.add(camid)

        print(cam_container, 'cam_container')
        return dataset
