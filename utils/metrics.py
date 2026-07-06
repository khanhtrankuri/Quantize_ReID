import torch
import numpy as np
import os
from utils.reranking import re_ranking

_DEFAULT_EVAL_QUERY_CHUNK_SIZE = 64
_DEFAULT_MAX_FULL_EVAL_ELEMENTS = 100_000_000


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        value = int(value)
    except ValueError:
        print("{} must be an integer, got {}; using {}".format(name, value, default))
        return default
    return max(1, value)


def _is_enabled(value):
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


def _euclidean_distance_tensor(qf, gf, gf_square=None):
    qf = qf.float()
    gf = gf.float()
    if gf_square is None:
        gf_square = torch.pow(gf, 2).sum(dim=1, keepdim=True).t()
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True) + gf_square
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat


def euclidean_distance(qf, gf):
    return _euclidean_distance_tensor(qf, gf).cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def _adjust_max_rank(num_g, max_rank):
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    return max_rank


def _eval_sorted_indices(indices, q_pids, g_pids, q_camids, g_camids, max_rank, query_offset=0):
    num_q = len(q_pids)
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    num_no_valid_q = 0

    for row_idx in range(indices.shape[0]):
        q_idx = query_offset + row_idx
        if q_idx >= num_q:
            break

        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[row_idx]

        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        orig_cmc = (g_pids[order] == q_pid).astype(np.int32)[keep]
        if not np.any(orig_cmc):
            num_no_valid_q += 1
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        cmc = cmc[:max_rank]
        if cmc.shape[0] < max_rank:
            cmc = np.pad(cmc, (0, max_rank - cmc.shape[0]), mode="edge")

        all_cmc.append(cmc)
        num_valid_q += 1.

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    return all_cmc, all_AP, num_valid_q, num_no_valid_q


def _finish_market1501_eval(all_cmc, all_AP, num_valid_q, num_no_valid_q, num_q):
    if num_no_valid_q > 0:
        print(
            "Market1501 eval: valid queries {}/{}; skipped {} queries with no cross-camera gallery match".format(
                int(num_valid_q), num_q, num_no_valid_q
            )
        )

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    max_rank = _adjust_max_rank(num_g, max_rank)
    indices = np.argsort(distmat, axis=1)
    all_cmc, all_AP, num_valid_q, num_no_valid_q = _eval_sorted_indices(
        indices, q_pids, g_pids, q_camids, g_camids, max_rank
    )
    return _finish_market1501_eval(all_cmc, all_AP, num_valid_q, num_no_valid_q, num_q)


def eval_func_chunked(qf, gf, q_pids, g_pids, q_camids, g_camids, max_rank=50, query_chunk_size=64):
    """Market1501 evaluation without materializing the full query x gallery matrix."""
    num_q = qf.shape[0]
    num_g = gf.shape[0]
    max_rank = _adjust_max_rank(num_g, max_rank)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    num_no_valid_q = 0

    qf = qf.float()
    gf = gf.float()
    gf_square = torch.pow(gf, 2).sum(dim=1, keepdim=True).t()
    num_chunks = (num_q + query_chunk_size - 1) // query_chunk_size

    for chunk_idx, start in enumerate(range(0, num_q, query_chunk_size), 1):
        end = min(start + query_chunk_size, num_q)
        distmat = _euclidean_distance_tensor(qf[start:end], gf, gf_square=gf_square).cpu().numpy()
        indices = np.argsort(distmat, axis=1)
        chunk_cmc, chunk_AP, chunk_valid_q, chunk_no_valid_q = _eval_sorted_indices(
            indices, q_pids, g_pids, q_camids, g_camids, max_rank, query_offset=start
        )
        all_cmc.extend(chunk_cmc)
        all_AP.extend(chunk_AP)
        num_valid_q += chunk_valid_q
        num_no_valid_q += chunk_no_valid_q

        if chunk_idx == 1 or chunk_idx == num_chunks or chunk_idx % 100 == 0:
            print("=> Eval chunk {}/{}: queries {}-{}".format(chunk_idx, num_chunks, start, end - 1))

    return _finish_market1501_eval(all_cmc, all_AP, num_valid_q, num_no_valid_q, num_q)


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False,
                 eval_query_chunk_size=None, max_full_eval_elements=None,
                 num_samples=None):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.num_samples = num_samples
        self.eval_query_chunk_size = _env_int(
            "REID_EVAL_QUERY_CHUNK_SIZE",
            eval_query_chunk_size or _DEFAULT_EVAL_QUERY_CHUNK_SIZE,
        )
        self.max_full_eval_elements = _env_int(
            "REID_EVAL_MAX_FULL_ELEMENTS",
            max_full_eval_elements or _DEFAULT_MAX_FULL_EVAL_ELEMENTS,
        )
        # REID_EVAL_HALF_FEATS=1 -> luu feature dang float16 thay vi float32,
        # giam ~50% RAM khi test set rat lon (vi du gop nhieu dataset lai).
        # Sai so do lam tron khi tinh khoang cach thuong khong dang ke voi
        # feature da chuan hoa (feat_norm=True).
        self.half_feats = _is_enabled(os.environ.get("REID_EVAL_HALF_FEATS", "0"))

    def reset(self):
        self.feats = None if self.num_samples is not None else []
        self.pids = None if self.num_samples is not None else []
        self.camids = None if self.num_samples is not None else []
        self._write_index = 0

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        feat = feat.detach().cpu()
        if self.half_feats:
            feat = feat.half()
        pid = np.asarray(pid)
        camid = np.asarray(camid)

        if self.num_samples is None:
            self.feats.append(feat)
            self.pids.extend(pid)
            self.camids.extend(camid)
            return

        end = self._write_index + feat.shape[0]
        if end > self.num_samples:
            raise RuntimeError(
                "Evaluator received more samples ({}) than expected ({})".format(end, self.num_samples)
            )

        if self.feats is None:
            self.feats = torch.empty((self.num_samples,) + tuple(feat.shape[1:]), dtype=feat.dtype)
            self.pids = np.empty((self.num_samples,), dtype=pid.dtype)
            self.camids = np.empty((self.num_samples,), dtype=camid.dtype)

        self.feats[self._write_index:end].copy_(feat)
        self.pids[self._write_index:end] = pid
        self.camids[self._write_index:end] = camid
        self._write_index = end

    def compute(self):  # called after each epoch
        if self.num_samples is None:
            feats = torch.cat(self.feats, dim=0)
            pids = self.pids
            camids = self.camids
        else:
            if self.feats is None:
                raise RuntimeError("Evaluator has no features to compute")
            if self._write_index != self.num_samples:
                print(
                    "Warning: evaluator expected {} samples but received {}".format(
                        self.num_samples, self._write_index
                    )
                )
            feats = self.feats[:self._write_index]
            pids = self.pids[:self._write_index]
            camids = self.camids[:self._write_index]
        self.feats = feats
        if _is_enabled(self.feat_norm):
            print("The test feature is normalized")
            # Chuyen ve float32 truoc khi normalize: mot so kernel CPU khong ho
            # tro (hoac kem chinh xac) tren tensor float16 (khi REID_EVAL_HALF_FEATS=1
            # duoc dung de giam RAM luc gom feature). Buffer float16 tich luy da
            # giai quyet phan RAM lon nhat (sustained trong suot vong lap trich
            # xuat feature); ban sao float32 o day chi ton tai tam thoi.
            feats = feats.float()
            feats.div_(torch.norm(feats, p=2, dim=1, keepdim=True).clamp_min_(1e-12))
            self.feats = feats
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(pids[:self.num_query])
        q_camids = np.asarray(camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(pids[self.num_query:])

        g_camids = np.asarray(camids[self.num_query:])
        num_pairs = qf.shape[0] * gf.shape[0]
        if self.reranking:
            total_samples = qf.shape[0] + gf.shape[0]
            if total_samples * total_samples > self.max_full_eval_elements:
                raise RuntimeError(
                    "Re-ranking needs a full {}x{} distance matrix. Disable reranking or use a smaller eval split.".format(
                        total_samples, total_samples
                    )
                )
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
            cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=self.max_rank)

        elif num_pairs > self.max_full_eval_elements:
            print(
                "=> Computing chunked DistMat eval for {} query x {} gallery pairs (chunk size {})".format(
                    qf.shape[0], gf.shape[0], self.eval_query_chunk_size
                )
            )
            distmat = None
            cmc, mAP = eval_func_chunked(
                qf,
                gf,
                q_pids,
                g_pids,
                q_camids,
                g_camids,
                max_rank=self.max_rank,
                query_chunk_size=self.eval_query_chunk_size,
            )
        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
            cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=self.max_rank)

        return cmc, mAP, distmat, pids, camids, qf, gf
