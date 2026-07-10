import argparse
import os
from collections import Counter, defaultdict


def parse_camid(path):
    name = os.path.basename(path)
    parts = name.split("_")
    if len(parts) < 3:
        raise ValueError("Cannot parse camera id from {}".format(path))
    return int(parts[2]) - 1


def parse_pid_from_path(path):
    name = os.path.basename(path)
    return int(name.split("_")[0])


def read_list(path):
    records = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rel_path, pid = line.split()
            records.append((rel_path, int(pid), parse_camid(rel_path)))
    return records


def scan_test_dir(test_dir):
    records = []
    for root, _, files in os.walk(test_dir):
        for filename in files:
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            abs_path = os.path.join(root, filename)
            rel_path = os.path.relpath(abs_path, test_dir)
            records.append((rel_path, parse_pid_from_path(rel_path), parse_camid(rel_path)))
    return sorted(records)


def write_list(path, records):
    with open(path, "w") as handle:
        for rel_path, pid, _ in records:
            handle.write("{} {}\n".format(rel_path, pid))


def summarize(records, label):
    pids = {pid for _, pid, _ in records}
    cams = Counter(camid for _, _, camid in records)
    print("{}: images={}, ids={}, cameras={}".format(label, len(records), len(pids), dict(cams)))


def build_split(records, train_records=None, drop_train_overlap=False):
    train_pids = {pid for _, pid, _ in train_records} if train_records else set()
    by_pid = defaultdict(list)
    for record in records:
        _, pid, _ = record
        if drop_train_overlap and pid in train_pids:
            continue
        by_pid[pid].append(record)

    query = []
    gallery = []
    invalid_single_camera_ids = []

    for pid in sorted(by_pid):
        pid_records = sorted(by_pid[pid])
        cams = defaultdict(list)
        for record in pid_records:
            cams[record[2]].append(record)

        if len(cams) < 2:
            invalid_single_camera_ids.append(pid)
            gallery.extend(pid_records)
            continue

        query_cam = sorted(cams.keys())[0]
        query_record = sorted(cams[query_cam])[0]
        query.append(query_record)
        gallery.extend(record for record in pid_records if record != query_record)

    return query, gallery, invalid_single_camera_ids


def audit_split(train, query, gallery):
    gallery_cams_by_pid = defaultdict(set)
    for _, pid, camid in gallery:
        gallery_cams_by_pid[pid].add(camid)

    valid = 0
    invalid = []
    for idx, (_, pid, qcam) in enumerate(query):
        gcams = gallery_cams_by_pid.get(pid, set())
        if any(gcam != qcam for gcam in gcams):
            valid += 1
        else:
            invalid.append((idx, pid, qcam, sorted(gcams)))

    train_pids = {pid for _, pid, _ in train} if train else set()
    query_pids = {pid for _, pid, _ in query}
    gallery_pids = {pid for _, pid, _ in gallery}
    print("valid cross-camera queries: {} / {}".format(valid, len(query)))
    print("invalid queries: {} / {}".format(len(invalid), len(query)))
    print("train/query ID overlap: {}".format(len(train_pids & query_pids)))
    print("train/gallery ID overlap: {}".format(len(train_pids & gallery_pids)))
    print("query/gallery ID overlap: {}".format(len(query_pids & gallery_pids)))
    print("first 20 invalid queries:")
    for item in invalid[:20]:
        print("idx={}, pid={}, qcam={}, gallery_cams={}".format(*item))


def main():
    parser = argparse.ArgumentParser(description="Audit or rebuild ReID query/gallery split lists")
    parser.add_argument("--dataset_dir", default="/mnt/data/khanhtl/ReID/MSMT17_datatt")
    parser.add_argument("--source", default="lists", choices=["lists", "test_dir"])
    parser.add_argument("--drop_train_overlap", action="store_true")
    parser.add_argument("--output_dir", default="", help="write rebuilt list_query.txt/list_gallery.txt here")
    args = parser.parse_args()

    train_records = []
    for name in ("list_train.txt", "list_val.txt"):
        path = os.path.join(args.dataset_dir, name)
        if os.path.exists(path):
            train_records.extend(read_list(path))

    if args.source == "test_dir":
        records = scan_test_dir(os.path.join(args.dataset_dir, "test"))
    else:
        records = []
        records.extend(read_list(os.path.join(args.dataset_dir, "list_query.txt")))
        records.extend(read_list(os.path.join(args.dataset_dir, "list_gallery.txt")))

    summarize(train_records, "train+val")
    summarize(records, "candidate test records")
    query, gallery, single_camera_ids = build_split(
        records,
        train_records=train_records,
        drop_train_overlap=args.drop_train_overlap,
    )
    summarize(query, "rebuilt query")
    summarize(gallery, "rebuilt gallery")
    print("ids with only one camera in candidate records: {}".format(len(single_camera_ids)))
    audit_split(train_records, query, gallery)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        write_list(os.path.join(args.output_dir, "list_query.txt"), query)
        write_list(os.path.join(args.output_dir, "list_gallery.txt"), gallery)
        print("Wrote rebuilt split lists to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
