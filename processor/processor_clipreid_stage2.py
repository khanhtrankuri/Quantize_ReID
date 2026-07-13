import logging
import os
import time
from collections import Counter, defaultdict
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
from model.clip.qat_layers import disable_qat_observers, enable_fake_quant, enable_qat_observers


def _cfg_enabled(value):
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return max(1, int(value))
    except ValueError:
        return default


def _rss_gib():
    try:
        with open("/proc/self/statm", "r") as statm:
            resident_pages = int(statm.readline().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3)
    except (OSError, IndexError, ValueError):
        return None


def _format_mem():
    parts = []
    rss = _rss_gib()
    if rss is not None:
        parts.append("rss={:.2f}GiB".format(rss))
    if torch.cuda.is_available():
        parts.append("cuda_alloc={:.2f}GiB".format(torch.cuda.memory_allocated() / (1024 ** 3)))
        parts.append("cuda_reserved={:.2f}GiB".format(torch.cuda.memory_reserved() / (1024 ** 3)))
    return ", ".join(parts) if parts else "memory=n/a"


def _log_eval_split_audit(logger, val_loader, num_query, invalid_preview=20):
    dataset = getattr(val_loader.dataset, "dataset", None)
    if dataset is None:
        logger.warning("Could not audit eval split: val_loader.dataset has no raw dataset attribute")
        return

    query = dataset[:num_query]
    gallery = dataset[num_query:]
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

    logger.info("=" * 80)
    logger.info("ReID eval split audit")
    logger.info("num query: {}".format(len(q_pids)))
    logger.info("num gallery: {}".format(len(g_pids)))
    logger.info("query IDs: {}".format(len(set(q_pids))))
    logger.info("gallery IDs: {}".format(len(set(g_pids))))
    logger.info("common IDs: {}".format(len(set(q_pids) & set(g_pids))))
    logger.info("valid cross-camera queries: {} / {}".format(valid, len(q_pids)))
    logger.info("invalid queries: {} / {}".format(len(invalid), len(q_pids)))
    logger.info("query camera distribution: {}".format(Counter(q_camids)))
    logger.info("gallery camera distribution: {}".format(Counter(g_camids)))
    logger.info("first {} invalid queries:".format(invalid_preview))
    for idx, pid, qcam, gallery_cams in invalid[:invalid_preview]:
        logger.info("idx={}, pid={}, qcam={}, gallery_cams={}".format(idx, pid, qcam, gallery_cams))
    if q_pids and float(valid) / float(len(q_pids)) < 0.8:
        logger.warning(
            "WARNING: Only {}/{} queries have valid cross-camera gallery matches. Metrics may be unreliable.".format(
                valid, len(q_pids)
            )
        )
    logger.info("=" * 80)


@torch.no_grad()
def _calibrate_qat(cfg, model, loader, device, logger):
    batches = int(getattr(cfg.MODEL.QAT, "CALIBRATION_BATCHES", 0))
    if not cfg.MODEL.QAT.ENABLED or batches <= 0:
        return
    qat_model = model.module if isinstance(model, nn.DataParallel) else model
    was_training = qat_model.training
    qat_model.eval()
    enable_qat_observers(qat_model)
    # Calibration observes true FP32 activations; fake quant starts in adaptation.
    from model.clip.qat_layers import disable_fake_quant
    disable_fake_quant(qat_model)
    seen = 0
    for image, _, camera, view in loader:
        image = image.to(device)
        camera = camera.to(device) if cfg.MODEL.SIE_CAMERA else None
        view = view.to(device) if cfg.MODEL.SIE_VIEW else None
        qat_model(x=image, cam_label=camera, view_label=view)
        seen += 1
        if seen >= batches:
            break
    enable_fake_quant(qat_model)
    smoothing = cfg.MODEL.QAT.SMOOTHING
    if bool(smoothing.ENABLED) and hasattr(qat_model, "image_encoder") and hasattr(qat_model.image_encoder, "transformer"):
        from model.clip.smoothing import smooth_transformer_blocks
        count = smooth_transformer_blocks(qat_model.image_encoder, alpha=float(smoothing.ALPHA))
        logger.info("Folded SmoothQuant scales into %d transformer groups (alpha=%s).", count, smoothing.ALPHA)
    if was_training:
        qat_model.train()
    logger.info("QAT calibration completed on %d batches; fake quant is now enabled.", seen)


def _distillation_loss(student_feat, teacher_feat, cfg):
    student = F.normalize(student_feat.float(), dim=1)
    teacher = F.normalize(teacher_feat.detach().float(), dim=1)
    feature = (1.0 - F.cosine_similarity(student, teacher, dim=1)).mean()
    relation = F.mse_loss(student @ student.t(), teacher @ teacher.t())
    if student.shape[0] > 1:
        diagonal = torch.eye(student.shape[0], device=student.device, dtype=torch.bool)
        distance = F.mse_loss((1.0 - student @ student.t())[~diagonal], (1.0 - teacher @ teacher.t())[~diagonal])
    else:
        distance = student.new_zeros(())
    terms = cfg.MODEL.QAT.DISTILLATION
    total = terms.FEATURE_WEIGHT * feature + terms.RELATION_WEIGHT * relation + terms.DISTANCE_WEIGHT * distance
    return total, feature, relation, distance

def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank, teacher_model=None):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(
        num_query,
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        num_samples=len(val_loader.dataset),
    )
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    if cfg.MODEL.QAT.ENABLED:
        qat_model = model.module if isinstance(model, nn.DataParallel) else model
        enable_qat_observers(qat_model)
        enable_fake_quant(qat_model)
        logger.info(
            "QAT enabled: fake quant active; observer freeze epoch: {}".format(
                cfg.MODEL.QAT.DISABLE_OBSERVER_EPOCH
            )
        )
        _calibrate_qat(cfg, model, train_loader_stage2, device, logger)
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    # train
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes-batch* (num_classes//batch)
    if left != 0 :
        i_ter = i_ter+1
    text_features = []
    with torch.no_grad():
        for i in range(i_ter):
            if i+1 != i_ter:
                l_list = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list = torch.arange(i*batch, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model(label = l_list, get_text = True)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        if cfg.MODEL.QAT.ENABLED and cfg.MODEL.QAT.DISABLE_OBSERVER_EPOCH > 0 and epoch >= cfg.MODEL.QAT.DISABLE_OBSERVER_EPOCH:
            qat_model = model.module if isinstance(model, nn.DataParallel) else model
            disable_qat_observers(qat_model)
            enable_fake_quant(qat_model)

        scheduler.step()

        model.train()
        if teacher_model is not None:
            teacher_model.eval()
        feature_distill_meter = AverageMeter()
        relation_distill_meter = AverageMeter()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            with amp.autocast(enabled=True):
                score, feat, image_features = model(x = img, label = target, cam_label=target_cam, view_label=target_view)
                logits = image_features @ text_features.t()
                loss = loss_fn(score, feat, target, target_cam, logits)
                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_feature = teacher_model(img, cam_label=target_cam, view_label=target_view)
                    if cfg.MODEL.QAT.DISTILLATION.USE_PRE_BN_FEATURE:
                        student_feature = torch.cat([feat[1], feat[2]], dim=1)
                    else:
                        student_feature = torch.cat([model.module.bottleneck(feat[1]) if isinstance(model, nn.DataParallel) else model.bottleneck(feat[1]),
                                                     model.module.bottleneck_proj(feat[2]) if isinstance(model, nn.DataParallel) else model.bottleneck_proj(feat[2])], dim=1)
                    distill, loss_feature, loss_relation, loss_distance = _distillation_loss(student_feature, teacher_feature, cfg)
                    loss = loss + distill

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            if teacher_model is not None:
                feature_distill_meter.update(loss_feature.item(), img.shape[0])
                relation_distill_meter.update(loss_relation.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, DistillFeature: {:.4f}, DistillRelation: {:.4f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_meter.avg, acc_meter.avg, feature_distill_meter.avg, relation_distill_meter.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            precision_tag = 'qat' if cfg.MODEL.QAT.ENABLED else 'fp32'
            checkpoint_name = '{}_{}_{}.pth'.format(cfg.MODEL.NAME, precision_tag, epoch)
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, checkpoint_name))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, checkpoint_name))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else: 
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else: 
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else: 
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else: 
                            target_view = None
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    num_total = len(val_loader.dataset)
    num_gallery = num_total - num_query
    max_full_eval_elements = _env_int("REID_EVAL_MAX_FULL_ELEMENTS", 100_000_000)
    reranking = _cfg_enabled(cfg.TEST.RE_RANKING)
    if reranking and (num_query + num_gallery) * (num_query + num_gallery) > max_full_eval_elements:
        logger.warning(
            "TEST.RE_RANKING=True needs a full {}x{} distance matrix; disabling re-ranking for this run to avoid RAM OOM.".format(
                num_query + num_gallery, num_query + num_gallery
            )
        )
        reranking = False

    logger.info(
        "Inference set: query={}, gallery={}, total={}, batches={}, batch_size={}, reranking={}, {}".format(
            num_query,
            num_gallery,
            num_total,
            len(val_loader),
            val_loader.batch_size,
            reranking,
            _format_mem(),
        )
    )
    _log_eval_split_audit(logger, val_loader, num_query)

    evaluator = R1_mAP_eval(
        num_query,
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=reranking,
        num_samples=num_total,
    )

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
    model.to(device)

    model.eval()

    log_period = max(1, _env_int("REID_INFER_LOG_PERIOD", 500))
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
        if (n_iter + 1) == 1 or (n_iter + 1) % log_period == 0 or (n_iter + 1) == len(val_loader):
            logger.info(
                "Inference progress: batch {}/{}, images {}, {}".format(
                    n_iter + 1,
                    len(val_loader),
                    min((n_iter + 1) * val_loader.batch_size, num_total),
                    _format_mem(),
                )
            )


    logger.info("Feature extraction complete; computing metrics, {}".format(_format_mem()))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4], mAP
