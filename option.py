import argparse

def args_parser():

    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=[
                            'cifar100',
                            'imagenet100', 'imagenet',], help="name of dataset")
    parser.add_argument('--img_size', type=int, default=224, help="size of images")
    parser.add_argument('--init_classes', type=int, default=10, help='number of data classes of first task')
    parser.add_argument('--task_size', type=int, default=10, help='number of data classes each task')
    parser.add_argument('--dataset_path', type=str, default='./data', help="the path of dataset")
    parser.add_argument('--batch_size', type=int, default=64, help='size of mini-batch')
    # CILformer settings
    parser.add_argument('--method', type=str, default='CILformer', help="name of method")
    parser.add_argument('--device', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--msa_blocks', type=int, default=5, help='the number of blocks of MSA')
    parser.add_argument('--model', type=str, default='vit_base', help="name of model")
    parser.add_argument('--model_depth', type=int, default=11, help="name of model")
    parser.add_argument('--model_path', type=str, default='./vit_base.pth', help="the path of model")

    parser.add_argument('--use_adapter', default=True,action='store_true')

    parser.add_argument('--ffn_option', type=str, default="parallel")
    parser.add_argument('--ffn_adapter_layernorm_option', type=str, default="none")
    parser.add_argument('--ffn_adapter_init_option', type=str, default="lora")
    parser.add_argument('--ffn_adapter_scalar', type=float, default=0.1)
    parser.add_argument('--ffn_num', type=int, default=64)

    # training settings
    parser.add_argument('--epochs', type=int, default=30, help='local epochs of each global round')
    parser.add_argument('--lr', type=float, default=0.0000625, help='learning rate')
    parser.add_argument('--warm_up', default=True)
    parser.add_argument('--optim', type=str, default='adam', help="name of dataset")
    parser.add_argument('--min_lr', type=float, default=0.00001, help='learning rate')
    # memory
    parser.add_argument('--resume', type=int, default=0, help='local epochs of each global round')
    parser.add_argument('--memory_size', type=int, default=2000, help='size of exemplar memory')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--distributed-memory', default=False, action='store_true',
                        help='Use different rehearsal memory per process.')
    parser.add_argument('--global-memory', default=False, action='store_false', dest='distributed_memory',
                        help='Use same rehearsal memory for all process.')
    parser.add_argument('--oversample-memory', default=1, type=int,
                        help='Amount of time we repeat the same rehearsal.')
    parser.add_argument('--oversample-memory-ft', default=1, type=int,
                        help='Amount of time we repeat the same rehearsal for finetuning, only for old classes not new classes.')
    parser.add_argument('--rehearsal-test-trsf', default=False, action='store_true',
                        help='Extract features without data augmentation.')
    parser.add_argument('--rehearsal-modes', default=1, type=int,
                        help='Select N on a single gpu, but with mem_size/N.')
    parser.add_argument('--fixed-memory', default=False, action='store_true',
                        help='Dont fully use memory when no all classes are seen as in Hou et al. 2019')
    parser.add_argument('--rehearsal', default="icarl_all",
                        choices=[
                            'random',
                            'closest_token', 'closest_all',
                            'icarl_token', 'icarl_all',
                            'furthest_token', 'furthest_all'
                        ],
                        help='Method to herd sample for rehearsal.')
    parser.add_argument('--sep-memory', default=False, action='store_true',
                        help='Dont merge memory w/ task dataset but keep it alongside')
    parser.add_argument('--replay-memory', default=0, type=int,
                        help='Replay memory according to Guido rule [NEED DOC]')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')  
    args = parser.parse_args()
    return args