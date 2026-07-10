from collections import Counter, defaultdict


def _split_pids(split):
    return {int(item[1]) for item in split}


def audit_reid_split(train, query, gallery, print_fn=print, invalid_preview=20):
    q_pids = [int(item[1]) for item in query]
    g_pids = [int(item[1]) for item in gallery]
    q_camids = [int(item[2]) for item in query]
    g_camids = [int(item[2]) for item in gallery]

    gallery_cams_by_pid = defaultdict(set)
    for pid, camid in zip(g_pids, g_camids):
        gallery_cams_by_pid[pid].add(camid)

    valid = 0
    invalid = []
    for idx, (pid, qcam) in enumerate(zip(q_pids, q_camids)):
        gallery_cams = gallery_cams_by_pid.get(pid, set())
        if any(gallery_camid != qcam for gallery_camid in gallery_cams):
            valid += 1
        else:
            invalid.append((idx, pid, qcam, sorted(gallery_cams)))

    query_pids = set(q_pids)
    gallery_pids = set(g_pids)
    train_pids = _split_pids(train) if train is not None else set()
    valid_ratio = float(valid) / float(len(q_pids)) if q_pids else 0.0

    print_fn("=" * 80)
    print_fn("ReID split audit")
    print_fn("num query: {}".format(len(q_pids)))
    print_fn("num gallery: {}".format(len(g_pids)))
    print_fn("query IDs: {}".format(len(query_pids)))
    print_fn("gallery IDs: {}".format(len(gallery_pids)))
    print_fn("common IDs: {}".format(len(query_pids & gallery_pids)))
    print_fn("valid cross-camera queries: {} / {}".format(valid, len(q_pids)))
    print_fn("invalid queries: {} / {}".format(len(invalid), len(q_pids)))
    print_fn("query camera distribution: {}".format(Counter(q_camids)))
    print_fn("gallery camera distribution: {}".format(Counter(g_camids)))
    print_fn("first {} invalid queries:".format(invalid_preview))
    for idx, pid, qcam, gallery_cams in invalid[:invalid_preview]:
        print_fn("idx={}, pid={}, qcam={}, gallery_cams={}".format(idx, pid, qcam, gallery_cams))

    if train is not None:
        train_query_overlap = len(train_pids & query_pids)
        train_gallery_overlap = len(train_pids & gallery_pids)
        query_gallery_overlap = len(query_pids & gallery_pids)
        print_fn("train/query ID overlap: {}".format(train_query_overlap))
        print_fn("train/gallery ID overlap: {}".format(train_gallery_overlap))
        print_fn("query/gallery ID overlap: {}".format(query_gallery_overlap))
        if train_query_overlap > 0 or train_gallery_overlap > 0:
            print_fn(
                "WARNING: Train IDs overlap query/gallery IDs. This can indicate data leakage unless the dataset "
                "intentionally uses disjoint label spaces after relabeling."
            )

    if q_pids and valid_ratio < 0.8:
        print_fn(
            "WARNING: Only {}/{} queries have valid cross-camera gallery matches. Metrics may be unreliable.".format(
                valid, len(q_pids)
            )
        )
    print_fn("=" * 80)

    return {
        "num_query": len(q_pids),
        "num_gallery": len(g_pids),
        "num_query_ids": len(query_pids),
        "num_gallery_ids": len(gallery_pids),
        "num_common_ids": len(query_pids & gallery_pids),
        "valid_queries": valid,
        "invalid_queries": len(invalid),
        "valid_ratio": valid_ratio,
        "train_query_overlap": len(train_pids & query_pids) if train is not None else None,
        "train_gallery_overlap": len(train_pids & gallery_pids) if train is not None else None,
    }
