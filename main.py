import copy
import os.path as osp
import time
import os
import torch.cuda
from train import Trainer
from option import args_parser
import warnings
from utils import *
warnings.filterwarnings("ignore")
from rehearsal import *
from dataset import build_dataset
from vit import vit_base_patch16_224,vit_small_patch16_224,vit_large_patch16_224
from model import Network
def train(args):
    # extractor
    path=args.model_path
    if args.model=='vit_base':
        feature_extractor=vit_base_patch16_224(args=args,depth=args.model_depth)
        feature_extractor.load_state_dict(torch.load(path)['model'],strict=False)
    elif args.model=='vit_small':
        feature_extractor=vit_small_patch16_224(depth=args.model_depth)
        feature_extractor.load_state_dict(torch.load('vit_small.pth')['model'],strict=False)
    # seed settings
    setup_seed(args.seed)
    args.device=torch.device("cuda:0")
    # model settings
    model = Network(feature_extractor,args).to(args.device)
    model_old = None
    # datasets
    train_dataset, args.nb_classes = build_dataset(is_train=True, args=args)
    test_dataset, _ = build_dataset(is_train=False, args=args)
    # memory               
    memory=Memory(args.memory_size,train_dataset.nb_classes,args.rehearsal,args.fixed_memory,args.rehearsal_modes)
    
    # incremental task
    args.num_tasks=args.nb_classes//args.task_size
    args.epochs_local=args.epochs//10
    args.epochs_global=10*args.num_tasks
    trainer= Trainer(model,args)
    # training log
    output_dir = osp.join('./training_log', args.method, 'seed' + str(args.seed))
    if not osp.exists(output_dir):
        os.system('mkdir -p ' + output_dir)
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    # model save
    model_global_dir = osp.join('./model', 'ckpt')
    if not osp.exists(model_global_dir):
        os.system('mkdir -p ' + model_global_dir)
    if not osp.exists(model_global_dir):
        os.mkdir(model_global_dir)
    args_param=vars(args)


    out_file = open(osp.join(output_dir, 'log_tar_' + str(args.task_size)+'_'+args.dataset+str(time.time())+'.txt'), 'w')
    log_str = 'method_{}, task_size_{}, learning_rate_{} dataset_{}'.format(args.method, args.task_size, args.lr, args.dataset)
    out_file.write(log_str +'\n'+ time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime(time.time()))+'\n')
    out_file.flush()

    out_file.write(str(args_param))
    out_file.flush()
    old_task_id = -1

    for ep_g in range(args.epochs_global):
        task_id = ep_g // 10

        if task_id != old_task_id and old_task_id != -1:
            # update old model

            model.eval()
            model_old = copy.deepcopy(model)
            # update model 
            model.incremental()
        
        if task_id !=old_task_id:
            # update dataset
            for id, dataset_train in enumerate(train_dataset):
                dataset_val=test_dataset[:id+1]
                if task_id==id:
                    break
            if task_id > 0 and memory is not None:
                dataset_memory = memory.get_dataset(dataset_train)

                previous_size = len(dataset_train)

                for _ in range(args.oversample_memory):
                    dataset_train.add_samples(*memory.get())
                print(f"{len(dataset_train) - previous_size} samples added from memory.")
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=8,
                pin_memory=True,
                drop_last=True,
            )

            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=int(1.5 * args.batch_size),
            )
            trainer.train_loader=data_loader_train

        print('task_id: {}'.format(task_id))
        # training
        model= trainer.train(ep_g, task_id, model_old)
        
        print(time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime(time.time())))
        acc_1,f1_1,recall_1,acc_5,f1_5,recall_5 =model_eval(model, data_loader_val, task_id, args.device,args,ep_g)
        log_str = 'Task: {}, Round: {} Acc1 = {:.2f}% f1_1 = {:.2f}% recall_1 = {:.2f}% Acc5 = {:.2f}% f1_5 = {:.2f}% recall_5 = {:.2f}% '.format(task_id, ep_g, acc_1,f1_1,recall_1,acc_5,f1_5,recall_5)
        out_file.write(log_str + time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime(time.time()))+ '\n')
        out_file.flush()
        print(time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime(time.time())))

        if ep_g%10==9:
            model.eval()
            # update memory
            task_set_to_rehearse = train_dataset[task_id]
            memory.add(task_set_to_rehearse, model, args.init_classes if task_id==0 else args.task_size)
            assert len(memory) <= args.memory_size, (len(memory), args.memory_size)

        torch.cuda.empty_cache()
        old_task_id = task_id

if __name__=="__main__":
    args = args_parser()

    train(args)
