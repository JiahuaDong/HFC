import torch
from sklearn.metrics import f1_score, recall_score
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from model import *
import random
import torch
import samplers

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
        
def get_loaders(dataset_train, dataset_val, args, finetuning=False):
    sampler_train, sampler_val = samplers.get_sampler(dataset_train, dataset_val, args)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=None if (finetuning and args.ft_no_sampling) else sampler_train,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=len(sampler_train) > args.batch_size,
    )

    loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    return loader_train, loader_val


def get_train_loaders(dataset_train, args, batch_size=None, drop_last=True):
    batch_size = batch_size or args.batch_size

    sampler_train = samplers.get_train_sampler(dataset_train, args)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=drop_last,
    )

    return loader_train

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def model_eval(model, test_loader, task_id, device,args,ep_g):
    model.eval()
    lb=np.arange(args.task_size * (task_id + 1))
    num_lb=lb.tolist()
    pre=[]
    label=[]
    pre_5=[]
    correct_1,correct_5, total = 0, 0, 0
    for step, (imgs, labels,indexs) in enumerate(test_loader):
        imgs, labels = imgs.cuda(device), labels.cuda(device)
        with torch.no_grad():
            outputs, _ = model(imgs)
            outputs=outputs.detach()
        predicts = torch.max(outputs, dim=1)[1]
        correct_1 += (predicts.cpu() == labels.cpu()).sum()
        total += len(labels)
        if len(num_lb)>5:
            maxk=max((1,5))
            y_resize=labels.view(-1,1)
            _, pred5=outputs.topk(maxk,1,True,True)
            index_l=torch.eq(pred5,y_resize).float()
            index_l=torch.sum(index_l,dim=1)
            pred=torch.where(index_l==1,labels,predicts)

            correct_5+=torch.eq(pred5.cpu(),y_resize.cpu()).sum()
            pre_5.append(pred.cpu())

        pre.append(predicts.cpu())
        label.append(labels.cpu())

    pre=torch.cat(pre,0)
    label=torch.cat(label,0)
    if len(num_lb)>5:
        pre_5 = torch.cat(pre_5, 0)
        f1_5 = f1_score(pre_5, label, labels=lb, average="macro") * 100
        recall_5 = recall_score(pre_5, label, labels=lb, average="macro") * 100
        acc_5 = 100 * correct_5 / total
    else:
        f1_5=100
        recall_5=100
        acc_5=100
    f1_1 = f1_score(pre, label, labels=lb, average="macro") *100
    recall_1 = recall_score(pre, label, labels=lb, average="macro") *100
    acc_1= 100 * correct_1 / total

    print('classification accuracy of global model at round %d: %.3f \n' % (ep_g, acc_1))
    print('classification accuracy of global model at round %d: %.3f \n' % (ep_g, f1_1))
    print('classification accuracy of global model at round %d: %.3f \n' % (ep_g, recall_1))
    print('classification accuracy of global model at round %d: %.3f \n' % (ep_g, acc_5))
    print('classification accuracy of global model at round %d: %.3f \n' % (ep_g, f1_5))
    print('classification accuracy of global model at round %d: %.3f \n' % (ep_g, recall_5))
    return acc_1,f1_1,recall_1,acc_5,f1_5,recall_5


