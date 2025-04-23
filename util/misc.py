# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import seaborn as sns
import numpy as np
import torch
import torch.distributed as dist
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from torch import nn
from torch._six import inf

from models import models_vit_2
from models.resnet import resnet50


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, checkpoint_name):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    # 新的检查点路径
    checkpoint_path = os.path.join(args.task, checkpoint_name)

    # 删除旧的检查点文件
    for file in os.listdir(args.task):
        # 假设旧文件包含 'checkpoint' 字样，并且文件格式相同
        if file.startswith('checkpoint') and file.endswith('.pth') and file != checkpoint_name:
            old_checkpoint_path = os.path.join(args.task, file)
            os.remove(old_checkpoint_path)
            print(f"Deleted old checkpoint: {old_checkpoint_path}")

    # 保存新权重
    if loss_scaler is not None:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }

        save_on_master(to_save, checkpoint_path)
        print(f"Saved new checkpoint: {checkpoint_path}")
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.task, tag="checkpoint-best", client_state=client_state)
        print("Saved checkpoint without loss scaler.")


def save_model_pretrain(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        print(model_without_ddp.state_dict().keys())
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def calculate_metrics(all_labels, pred_labels, probs, num_classes):
    # 计算各项指标
    accuracy = accuracy_score(all_labels, pred_labels)

    # 处理多分类AUC
    try:
        auc_roc = roc_auc_score(all_labels, probs, multi_class='ovr', average='macro')
    except ValueError as e:
        print("AUC计算错误:", e)
        auc_roc = None

    # 混淆矩阵
    cm = confusion_matrix(all_labels, pred_labels)

    return {
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'confusion_matrix': cm
    }


def validate_multi_auc_accuracy(model, dataloader, device, num_classes, criterion):
    model.eval()
    patient_predictions = defaultdict(list)
    patient_labels = {}
    running_loss = 0.0  # 初始化验证损失

    with torch.no_grad():
        for cfp_image, oct_image, labels, patient_ids in dataloader:
            cfp_image, oct_image = cfp_image.to(device), oct_image.to(device)
            labels = labels.to(device)  # 将标签移动到设备上

            # 模型预测
            output = model(cfp_image, oct_image)
            loss = criterion(output, labels)
            running_loss += loss.item() * cfp_image.size(0)

            probs = torch.softmax(output, dim=1).cpu().numpy()

            # 将批次中的每个预测存储到对应的患者ID中
            for prob, label, patient_id in zip(probs, labels.cpu().numpy(), patient_ids):
                patient_predictions[patient_id].append(prob)
                if patient_id not in patient_labels:
                    patient_labels[patient_id] = label

    all_patient_probs = []
    all_labels = []
    pred_labels = []

    # 逐个患者处理预测概率
    for patient_id, probs_list in patient_predictions.items():
        probs_list = np.squeeze(probs_list)

        # 选择具有最高置信度的一组概率
        if probs_list.ndim == 2:
            max_probs = probs_list[np.argmax(np.max(probs_list, axis=1))]
        elif probs_list.ndim == 1:
            max_probs = probs_list
        else:
            print(f"跳过异常数据: {patient_id}, 形状: {probs_list.shape}")
            continue

        all_patient_probs.append(max_probs)
        all_labels.append(patient_labels[patient_id])
        pred_labels.append(np.argmax(max_probs))

    all_patient_probs = np.array(all_patient_probs)
    all_labels = np.array(all_labels)

    # 确保 all_patient_probs 的列数和类别数一致
    if all_patient_probs.shape[1] != num_classes:
        all_patient_probs = np.pad(all_patient_probs, ((0, 0), (0, num_classes - all_patient_probs.shape[1])),
                                   'constant')

    # 计算准确率
    accuracy = accuracy_score(all_labels, pred_labels)

    # 计算AUC-ROC
    try:
        auc_roc = roc_auc_score(all_labels, all_patient_probs, multi_class='ovr', average='macro')
    except ValueError as e:
        print("Error calculating AUC-ROC:", e)
        auc_roc = None

    # 计算平均验证损失
    val_loss = running_loss / len(dataloader.dataset)

    print(f"Patient-based Validation Loss: {val_loss:.4f}")
    print(f"Patient-based Validation Accuracy: {accuracy:.4f}")
    if auc_roc is not None:
        print(f"Patient-based Validation AUC-ROC: {auc_roc:.4f}")

    return auc_roc, accuracy, all_labels, pred_labels, val_loss

def validate_model_auc_accuracy(model, val_loader, device, criterion):
    model.eval()  # 设置模型为评估模式
    all_preds = []
    all_labels = []
    pred_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)  # 假设两个加载器的标签是相同的


            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)


            probs = torch.nn.functional.softmax(outputs, dim=1)

            _, predictions = torch.max(outputs, 1)

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predictions.cpu().numpy())

    # 计算AUC-ROC
    auc_roc_score = roc_auc_score(all_labels, all_preds, multi_class='ovr',average='macro')  # 使用正确的概率列表
    val_loss = running_loss / len(val_loader.dataset)

    print(f"Patient-based Validation Loss: {val_loss:.4f}")
    print(f'Validation AUC-ROC: {auc_roc_score:.4f}')

    # 计算正确率
    accuracy = accuracy_score(all_labels, pred_labels)
    print(f'Validation Accuracy: {accuracy:.4f}')

    return auc_roc_score, accuracy, all_labels, pred_labels, val_loss


def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()


def load_resnet_model(weights_path, num_classes=6):
    # 加载ResNet50模型
    model = resnet50()
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, num_classes)

    # 加载预训练权重
    assert os.path.exists(weights_path), "Weights file does not exist."
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model


def load_retfound_model(model_name, checkpoint_path, img_size=224, num_classes=6, drop_path_rate=0.1,
                        global_pool=True):
    # 构建模型
    model = models_vit_2.__dict__[model_name](
        img_size=img_size,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        global_pool=global_pool,
    )

    # 加载预训练权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # 将模型移动到指定的设备上
    # model.to('cuda')

    return model
