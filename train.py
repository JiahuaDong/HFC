import math
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optim
from model import transformer_freeze
from utils import *
def get_one_hot(target, num_class, device):
    one_hot = torch.zeros(target.shape[0], num_class).to(device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)

    return one_hot
class InfiniteLoader:
    def __init__(self, loader):
        self.loader = loader
        self.reset()

    def reset(self):
        self.it = iter(self.loader)

    def get(self):
        try:
            return next(self.it)
        except StopIteration:
            self.reset()
            return self.get()

class Trainer:

    def __init__(self,model,args):

        super(Trainer, self).__init__()
        self.epochs = args.epochs_local
        self.learning_rate = args.lr
        self.model= model
        self.args= args
        self.numclass = 0
        self.learned_numclass = 0
        self.learned_classes = []
        self.old_model = None

        self.batchsize = args.batch_size
        self.task_size = args.task_size

        self.train_loader = None
        self.current_class = None
        self.last_class = None
        self.task_id_old = -1
        self.device = args.device
        self.freeze=args.msa_blocks
        self.exemplar_set=None

    def beforeTrain(self, task_id_new):
        if task_id_new != self.task_id_old:
            self.task_id_old = task_id_new
            self.numclass = self.args.init_classes+self.task_size * task_id_new

            if self.current_class is not None:
               self.last_class = self.current_class
            if task_id_new==0:
                self.current_class = [x for x in range(0, self.numclass)]
            else:
                self.current_class = [x for x in range(self.numclass - self.task_size, self.numclass)]
            print(self.current_class)
       
            if self.last_class is not None:
               self.learned_numclass += len(self.last_class)
               self.learned_classes += self.last_class
       
        self.model.train()
    
    # train model
    def train(self, ep_g, task_id_new, model_old):
        self.beforeTrain(task_id_new)
        self.model=self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad=True
           
        lr_init=self.args.lr
        epochs=self.args.epochs_local
        transformer_freeze(self.model.backbone,self.freeze)
            
        backbone=list(map(id,self.model.backbone.parameters()))
        fc=filter(lambda p:id(p) not in backbone,self.model.parameters())

        if model_old is not None:
            self.old_model = model_old
            self.old_model= self.old_model.to(self.device)
            print('load old model')
            self.old_model.eval()

        for epoch in range(epochs):

            if self.args.optim=='sgd':
                if ep_g%10<7:
                    lr=self.args.lr
                elif ep_g%10<9:
                    lr=self.args.lr/10
                else:
                    lr=self.args.lr/100
                if self.args.warm_up and ep_g%10==0:
                    lr=lr_init*(epoch+1)/(epochs+1)
                opt = optim.SGD([{'params':[p for n,p in list(self.model.backbone.named_parameters())],'lr': lr},
                                {'params': fc, 'lr': lr*5}], lr=lr*5, weight_decay=0.00001)
            else:
                step=epoch+(ep_g%10)*self.args.epochs_local
                lr = (math.cos(step /(10* self.args.epochs_local) * math.pi) + 1) * 0.5
                lr = self.args.min_lr + lr * ( self.args.lr- self.args.min_lr)
                lr = max(self.args.min_lr, lr)

                if self.args.warm_up and ep_g%10==0:
                    lr=lr_init*(epoch+1)/(epochs+1)
                opt = optim.AdamW([{'params':[p for n,p in list(self.model.backbone.named_parameters())],'lr': lr},
                    {'params': fc, 'lr': lr*5}], lr=lr*5, weight_decay=0.00001)
            print(opt.state_dict()['param_groups'][0]['lr'])
            print(opt.state_dict()['param_groups'][1]['lr'])
            for step, (images, labels,indexs) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                loss_value = self.compute_loss(images, labels)
                opt.zero_grad()
                loss_value.backward()
                opt.step()
            print(loss_value)
        return self.model
    # Our loss
    def compute_loss(self, images, labels):

        targets = get_one_hot(labels, self.numclass, self.device)
        output,features = self.model(images)

        output, targets = output.to(self.device), targets.to(self.device)
        if self.old_model == None:
            loss_cur = F.binary_cross_entropy_with_logits(output, targets)
            return loss_cur
        else:
            w1,w2 = self.gradient_balanced_weight(output, labels)
            # w1 = w1.clone().fill_(1.)
            # w2 = w2.clone().fill_(1.)
            # cross-entropy loss
            loss_cur = torch.mean(w1*F.binary_cross_entropy_with_logits(output, targets, reduction='none'))
            
            # distillation loss
            distill_targets = targets.clone()
            old_output,old_features = self.old_model(images)
            old_target = torch.sigmoid(old_output)
            old_task_size = old_target.shape[1]
            distill_targets[..., :old_task_size] = old_target
            loss_old = 0.5*self.compute_distri_dis(labels,output,distill_targets,w2)
            loss_old += torch.mean(w2*F.binary_cross_entropy_with_logits(output, distill_targets, reduction='none'))
            
            return loss_cur+loss_old

   
    def compute_distri_dis(self,labels,input,old_target,weight):
        classes=list(range(self.numclass))
        num=0
        for i in classes:
            ids = labels.view(-1, 1)
            ids = torch.where(ids != i, ids, ids.clone().fill_(-1))
            index = torch.eq(ids, -1).float()
   
            if index.sum() != 0:
                if num==0:
                    outputs = torch.div(torch.sum(input * index,dim=0), index.sum()).view(1,-1)
                    distill_targets=torch.div(torch.sum(old_target * index,dim=0), index.sum()).view(1,-1)
                    weights=torch.div(torch.sum(weight * index,dim=0), index.sum()).view(1,-1)
                else:
                    output = torch.div(torch.sum(input * index,dim=0), index.sum()).view(1,-1)
                    distill_target=torch.div(torch.sum(old_target * index,dim=0), index.sum()).view(1,-1)
                    w=torch.div(torch.sum(weight * index,dim=0), index.sum()).view(1,-1)
                    outputs=torch.cat((outputs,output),0)
                    distill_targets=torch.cat((distill_targets,distill_target),0)
                    weights=torch.cat((weights,w),0)
                num+=1
        loss=torch.mean(weights*F.binary_cross_entropy_with_logits(outputs, distill_targets, reduction='none'))
        return loss

    def gradient_balanced_weight(self, output, label):
        pred = torch.softmax(output,dim=1)
        N, C = pred.size(0), pred.size(1)
        class_mask = pred.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = label.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        target = get_one_hot(label, self.numclass, self.device)
        g = torch.abs(pred.detach() - target)
        if self.task_id_old > 0:
           z=torch.div(output.shape[1]-self.task_size,output.shape[1])
           y=g.clone().fill_(1)
           g=torch.pow(g,z)
           g=torch.log(g+y)
        g = (g * class_mask).sum(1).view(-1, 1)

        for i in self.learned_classes:
            ids = torch.where(ids != i, ids, ids.clone().fill_(-1))
        index_new = torch.ne(ids, -1).float()
        index_old = torch.eq(ids, -1).float()
        if index_new.sum() != 0:
            w1 = (1-z)* torch.clamp(torch.div(g * index_new, (g * index_new).sum() / index_new.sum()), 0.5, 1.0)*index_new
            w2 =  z * g.clone().fill_(0.8)*index_new
        else:
            w1 = g.clone().fill_(0.)
            w2 = g.clone().fill_(0.)
                
        for i in range(self.task_id_old):
            ids = label.view(-1, 1)
            
            task = self.args.init_classes+ i *self.task_size
            if i==0:
                classes = [n for n in self.learned_classes if n >=0 and n < task]
            else:
                classes = [n for n in self.learned_classes if n >=task-self.task_size and n < task]
            for j in classes:
                ids = torch.where(ids != j, ids, ids.clone().fill_(-1))
            index = torch.eq(ids, -1).float()
            if index.sum() != 0:
                w1 += torch.sqrt(1-z)* torch.clamp(torch.div(g * index, (g * index).sum() / index.sum()), 0.5, 1.2)*index
                w2 += z*torch.clamp(torch.div(g * index, (g * index).sum() / index.sum()), 0.5, 1.2)*index
            else:
                w1 += g.clone().fill_(0.)
                w2 += g.clone().fill_(0.)  
        return w1, w2
    