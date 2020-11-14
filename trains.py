import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from Pytorch.photo_detection.photo_detection_YOLOv3.PP_YOLO.networks.pp_yolos import ResNet50
from Pytorch.photo_detection.photo_detection_YOLOv3.PP_YOLO.dataloader import yolo_dataset_collate, YoloDataset
from Pytorch.photo_detection.photo_detection_YOLOv3.PP_YOLO.networks.yolo_trainings import YOLOLoss
from Pytorch.photo_detection.photo_detection_YOLOv3.PP_YOLO.utils.utils import EMA
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
from ranger_adabelief import RangerAdaBelief
device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def args_parse():
    parser = argparse.ArgumentParser(description="训练的参数")
    parser.add_argument('-input_size', default=(416, 416), type=int, help='input image size', dest='input_size')
    parser.add_argument('-anchors', default="./model_data/anchors.txt", type=str, help='anchor file')
    parser.add_argument('-classes', default="./model_data/class_name.txt", type=str, help='classes name')
    parser.add_argument('-annotation', default="./my_train.txt", type=str, help='annotation file')
    parser.add_argument('-model', default="./Pretraining_models/Epoch22-Total_Loss18.8616-Val_Loss18.8762.pth", type=str, help='model file')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-batch_size', default=32, type=int, help='train data batch size')
    parser.add_argument('-epochs', default=50, type=int, help='train epoch size')
    args = parser.parse_args()
    return args


def parse_lines(path):  # loads the lines
    with open(path) as f:
        lines_name = f.readlines()
    return lines_name


def get_classes(path):  # loads the classes
    class_names = parse_lines(path)
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(path):
    with open(path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


def train():
    args = args_parse()
    anchors = get_anchors(args.anchors)  # get anchors
    num_classes = len(get_classes(args.classes))  # get classes
    # 绘制模型
    models = ResNet50(len(anchors[0]), num_classes)

    # 使用预训练模型，如果显卡不够大的话可以使用预训练模型来微调
    print("Load pretrained model into state dict...")
    model_list = models.state_dict()
    pretrained_dict = torch.load(args.model, map_location=device)  # Load pretrained model
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_list}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_list[k]) == np.shape(v)}

    pretrained_dicts = {}  # 将pretrained_dict里不属于pretrained_dict的键剔除掉
    for k, v in pretrained_dict.items():
        # print(k)
        if np.shape(model_list[k]) == np.shape(v):  # # 用shape可以迅速的读取矩阵的形状
            # print(v)
            pretrained_dicts[k] = v

    model_list.update(pretrained_dicts)  # 把pretrained_dicts的键值对更新到model_list
    models.load_state_dict(model_list)
    print("Finished!")

    model = models.cuda()

    yolo_losses = []  # creat loss function
    input_shape = args.input_size
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(anchors, [-1, 2]), num_classes, (input_shape[1], input_shape[0]), label_smooth=0.1, cuda=True))

    val_slits = 0.1  # 训练集验证集分配
    lines = parse_lines(args.annotation)
    np.random.seed(5050)
    np.random.shuffle(lines)
    np.random.seed(None)
    val_num = int(len(lines) * val_slits)
    train_num = len(lines) - val_num

    # optimizer = torch.optim.Adam(model.parameters(), args.lr)  # 优化器
    optimizer = RangerAdaBelief(model.parameters(), lr=args.lr, eps=1e-12, betas=(0.9, 0.999))  # 优化器
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)  # 学习率余弦退火衰减
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)

    batch_size = args.batch_size
    train_dataset = YoloDataset(lines[:train_num], (input_shape[0], input_shape[1]))  # 进行数据处理
    val_dataset = YoloDataset(lines[train_num:], (input_shape[0], input_shape[1]))
    train = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
    val = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)

    train_epoch_size = train_num // batch_size
    val_epoch_size = val_num // batch_size

    for param in models.backbone.parameters():  # 冻结部分网络
        param.requires_grad = False
    # init 初始化
    ema = EMA(0.9998)

    # register 注册
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    # update 更新
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.update(name, param.data)

    writer = SummaryWriter(log_dir="./loges", flush_secs=60)  # 进行训练可视化
    epochs = args.epochs
    for epoch in range(0, epochs):
        total_loss = 0
        val_loss = 0
        start = time.time()
        model.train()
        with tqdm(total=train_epoch_size, desc=f"Epoch {epoch + 1} / {epochs}") as pbar:  # 进行训练
            for i, (data, targets) in enumerate(train):
                if i >= train_epoch_size:
                    break
                images = torch.from_numpy(data).cuda()
                targets = [torch.from_numpy(ann) for ann in targets]

                optimizer.zero_grad()
                outputs = model(images)

                losses = []
                for i in range(3):
                    loss_item = yolo_losses[i](outputs[i], targets)
                    losses.append(loss_item[0])
                loss = sum(losses)
                loss.backward()
                optimizer.step()

                total_loss += loss
                spend_time = time.time() - start

                def get_lr(optimizer):
                    for param_group in optimizer.param_groups:
                        return param_group['lr']

                dicts = {"step/s": spend_time, "lr": get_lr(optimizer), "total_loss": total_loss.item() / (i + 1)}
                pbar.set_postfix(dicts)  # 进度条右提示
                pbar.update(1)

                # 将loss写入tensorboard，每一步都写
                writer.add_scalar('Train_loss', loss, (epoch * train_epoch_size + i))

        start_time = time.time()
        model.eval()
        print("Start validation")
        with tqdm(total=val_epoch_size, desc=f'Epoch {epoch + 1} / {epochs}') as pbar:
            for i, (val_data, targets_val) in enumerate(val):
                if i >= val_epoch_size:
                    break
                with torch.no_grad():
                    images_val = torch.from_numpy(val_data).to(device)
                    targets_val = [torch.from_numpy(ann) for ann in targets_val]
                    optimizer.zero_grad()
                    outputs = model(images_val)

                    losses = []
                    for i in range(3):
                        loss_item = yolo_losses[i](outputs[i], targets_val)
                        losses.append(loss_item[0])
                    loss = sum(losses)
                    val_loss += loss

                spend_time = time.time() - start_time
                dicts = {"step/s": spend_time, "val_loss": val_loss.item() / (i + 1)}
                pbar.set_postfix(dicts)  # 进度条右提示
                pbar.update(1)

                # 将loss写入tensorboard，每个世代保存一次
                writer.add_scalar('Val_loss', val_loss / (val_epoch_size + 1), epoch)
                writer.close()

        print('Finish Validation')
        print('Epoch:' + str(epoch + 1) + '/' + str(epochs))
        print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (train_epoch_size + 1), val_loss / (val_epoch_size + 1)))

        print('Saving state, iter:', str(epoch + 1))
        torch.save(models.state_dict(), './models/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % ((epoch + 1), total_loss / (train_epoch_size + 1), val_loss / (val_epoch_size + 1)))
        lr_scheduler.step()


if __name__ == "__main__":
    train()
