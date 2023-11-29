import os
import json

from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder, default_loader
from continuum.datasets import CIFAR100 ,ImageFolderDataset,ImageNet100
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from continuum import ClassIncremental
""" Stanford Cars (Car) Dataset
Created: Nov 15,2019 - Yuchong Gu
Revised: Nov 15,2019 - Yuchong Gu
"""
import os
from PIL import Image
import pickle
import numpy as np
import torch
class ImageNet1000(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")
        return super().get_data()

class CarsDataset(Dataset):
    """
    # Description:
        Dataset for retrieving Stanford Cars images and labels
    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image
        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset
        __len__(self):                  returns the length of dataset
    """

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.phase = 'train' if train else 'test'
        # self.resize = resize
        self.num_classes = 196

        self.images = []
        self.labels = []

        list_path = os.path.join(root, 'cars_anno.pkl')

        list_mat = pickle.load(open(list_path, 'rb'))
        num_inst = len(list_mat['annotations']['relative_im_path'][0])
        for i in range(num_inst):
            if self.phase == 'train' and list_mat['annotations']['test'][0][i].item() == 0:
                path = list_mat['annotations']['relative_im_path'][0][i].item()
                label = list_mat['annotations']['class'][0][i].item()
                self.images.append(path)
                self.labels.append(label)
            elif self.phase != 'train' and list_mat['annotations']['test'][0][i].item() == 1:
                path = list_mat['annotations']['relative_im_path'][0][i].item()
                label = list_mat['annotations']['class'][0][i].item()
                self.images.append(path)
                self.labels.append(label)

        print('Car Dataset with {} instances for {} phase'.format(len(self.images), self.phase))

        # transform
        self.transform = transform

    def __getitem__(self, item):
        # image
        image = Image.open(os.path.join(self.root, self.images[item])).convert('RGB')  # (C, H, W)
        image = self.transform(image)

        # return image and label
        return image, self.labels[item] - 1  # count begin from zero

    def __len__(self):
        return len(self.images)


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.dataset.lower() == 'cifar100':
        dataset = CIFAR100(args.dataset_path, train=is_train, download=True)
    elif args.dataset.lower() == 'imagenet100':
        dataset = ImageNet100(
            args.dataset_path, train=is_train,
            data_subset=os.path.join('./imagenet100_splits', "train_100.txt" if is_train else "val_100.txt")
        )
    elif args.dataset.lower() == 'imagenet':
        dataset = ImageNet1000(args.dataset_path, train=is_train)
    else:
        raise ValueError(f'Unknown dataset {args.data_set}.')

    scenario = ClassIncremental(
        dataset,
        initial_increment=args.init_classes,
        increment=args.task_size,
        transformations=transform.transforms,
        class_order=get_class_order(args.dataset)
    )
    print(get_class_order(args.dataset))
    nb_classes = scenario.nb_classes

    return scenario, nb_classes

def build_transform(is_train, args, infer_no_resize=False):
    if hasattr(args, 'arch'):
        if 'cait' in args.arch and not is_train:
            print('# using cait eval transform')
            transformations = {}
            transformations= transforms.Compose(
                [transforms.Resize(args.input_size, interpolation=3),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
            return transformations
    
    if infer_no_resize:
        print('# using cait eval transform')
        transformations = {}
        transformations= transforms.Compose(
            [transforms.Resize(args.input_size, interpolation=3),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        return transformations

    resize_im = args.img_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
                input_size=args.img_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
            )


        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.img_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.img_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.img_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def get_class_order(dataset):
    if dataset.lower() == 'cifar100':
        class_order=[87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 
        14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 
        57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 
        75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 
        40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39]
    elif dataset.lower() == 'imagenet100':
        class_order=[68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 
        43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 
        44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97, 2, 64, 66, 
        42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 
        69, 36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33]
    elif dataset.lower() == 'imagenet':
        class_order=[54, 7, 894, 512, 126, 337, 988, 11, 284, 493, 133, 783, 192, 979, 
        622, 215, 240, 548, 238, 419, 274, 108, 928, 856, 494, 836, 473, 650, 85, 262, 
        508, 590, 390, 174, 637, 288, 658, 219, 912, 142, 852, 160, 704, 289, 123, 323, 
        600, 542, 999, 634, 391, 761, 490, 842, 127, 850, 665, 990, 597, 722, 748, 14, 
        77, 437, 394, 859, 279, 539, 75, 466, 886, 312, 303, 62, 966, 413, 959, 782, 509, 
        400, 471, 632, 275, 730, 105, 523, 224, 186, 478, 507, 470, 906, 699, 989, 324, 
        812, 260, 911, 446, 44, 765, 759, 67, 36, 5, 30, 184, 797, 159, 741, 954, 465, 533, 
        585, 150, 101, 897, 363, 818, 620, 824, 154, 956, 176, 588, 986, 172, 223, 461, 94, 
        141, 621, 659, 360, 136, 578, 163, 427, 70, 226, 925, 596, 336, 412, 731, 755, 381, 
        810, 69, 898, 310, 120, 752, 93, 39, 326, 537, 905, 448, 347, 51, 615, 601, 229, 947, 
        348, 220, 949, 972, 73, 913, 522, 193, 753, 921, 257, 957, 691, 155, 820, 584, 948, 
        92, 582, 89, 379, 392, 64, 904, 169, 216, 694, 103, 410, 374, 515, 484, 624, 409, 156, 
        455, 846, 344, 371, 468, 844, 276, 740, 562, 503, 831, 516, 663, 630, 763, 456, 179, 
        996, 936, 248, 333, 941, 63, 738, 802, 372, 828, 74, 540, 299, 750, 335, 177, 822, 
        643, 593, 800, 459, 580, 933, 306, 378, 76, 227, 426, 403, 322, 321, 808, 393, 27, 
        200, 764, 651, 244, 479, 3, 415, 23, 964, 671, 195, 569, 917, 611, 644, 707, 355, 
        855, 8, 534, 657, 571, 811, 681, 543, 313, 129, 978, 592, 573, 128, 243, 520, 887, 
        892, 696, 26, 551, 168, 71, 398, 778, 529, 526, 792, 868, 266, 443, 24, 57, 15, 871, 
        678, 745, 845, 208, 188, 674, 175, 406, 421, 833, 106, 994, 815, 581, 676, 49, 619,
         217, 631, 934, 932, 568, 353, 863, 827, 425, 420, 99, 823, 113, 974, 438, 874, 343, 
         118, 340, 472, 552, 937, 0, 10, 675, 316, 879, 561, 387, 726, 255, 407, 56, 927, 655, 
         809, 839, 640, 297, 34, 497, 210, 606, 971, 589, 138, 263, 587, 993, 973, 382, 572, 
         735, 535, 139, 524, 314, 463, 895, 376, 939, 157, 858, 457, 935, 183, 114, 903, 767, 
         666, 22, 525, 902, 233, 250, 825, 79, 843, 221, 214, 205, 166, 431, 860, 292, 976, 739, 
         899, 475, 242, 961, 531, 110, 769, 55, 701, 532, 586, 729, 253, 486, 787, 774, 165, 627,
          32, 291, 962, 922, 222, 705, 454, 356, 445, 746, 776, 404, 950, 241, 452, 245, 487, 706,
           2, 137, 6, 98, 647, 50, 91, 202, 556, 38, 68, 649, 258, 345, 361, 464, 514, 958, 504, 
           826, 668, 880, 28, 920, 918, 339, 315, 320, 768, 201, 733, 575, 781, 864, 617, 171, 795, 
           132, 145, 368, 147, 327, 713, 688, 848, 690, 975, 354, 853, 148, 648, 300, 436, 780, 
           693, 682, 246, 449, 492, 162, 97, 59, 357, 198, 519, 90, 236, 375, 359, 230, 476, 784, 
           117, 940, 396, 849, 102, 122, 282, 181, 130, 467, 88, 271, 793, 151, 847, 914, 42, 834,
            521, 121, 29, 806, 607, 510, 837, 301, 669, 78, 256, 474, 840, 52, 505, 547, 641, 987,
             801, 629, 491, 605, 112, 429, 401, 742, 528, 87, 442, 910, 638, 785, 264, 711, 369, 
             428, 805, 744, 380, 725, 480, 318, 997, 153, 384, 252, 985, 538, 654, 388, 100, 432, 
             832, 565, 908, 367, 591, 294, 272, 231, 213, 196, 743, 817, 433, 328, 970, 969, 4, 
             613, 182, 685, 724, 915, 311, 931, 865, 86, 119, 203, 268, 718, 317, 926, 269, 161, 
             209, 807, 645, 513, 261, 518, 305, 758, 872, 58, 65, 146, 395, 481, 747, 41, 283, 204, 
             564, 185, 777, 33, 500, 609, 286, 567, 80, 228, 683, 757, 942, 134, 673, 616, 960, 450, 
             350, 544, 830, 736, 170, 679, 838, 819, 485, 430, 190, 566, 511, 482, 232, 527, 411, 
             560, 281, 342, 614, 662, 47, 771, 861, 692, 686, 277, 373, 16, 946, 265, 35, 9, 884, 
             909, 610, 358, 18, 737, 977, 677, 803, 595, 135, 458, 12, 46, 418, 599, 187, 107, 992, 
             770, 298, 104, 351, 893, 698, 929, 502, 273, 20, 96, 791, 636, 708, 267, 867, 772, 604, 
             618, 346, 330, 554, 816, 664, 716, 189, 31, 721, 712, 397, 43, 943, 804, 296, 109, 576, 
             869, 955, 17, 506, 963, 786, 720, 628, 779, 982, 633, 891, 734, 980, 386, 365, 794, 325, 
             841, 878, 370, 695, 293, 951, 66, 594, 717, 116, 488, 796, 983, 646, 499, 53, 1, 603, 45, 
             424, 875, 254, 237, 199, 414, 307, 362, 557, 866, 341, 19, 965, 143, 555, 687, 235, 790, 
             125, 173, 364, 882, 727, 728, 563, 495, 21, 558, 709, 719, 877, 352, 83, 998, 991, 469, 
             967, 760, 498, 814, 612, 715, 290, 72, 131, 259, 441, 924, 773, 48, 625, 501, 440, 82, 
             684, 862, 574, 309, 408, 680, 623, 439, 180, 652, 968, 889, 334, 61, 766, 399, 598, 798, 
             653, 930, 149, 249, 890, 308, 881, 40, 835, 577, 422, 703, 813, 857, 995, 602, 583, 167, 
             670, 212, 751, 496, 608, 84, 639, 579, 178, 489, 37, 197, 789, 530, 111, 876, 570, 700, 
             444, 287, 366, 883, 385, 536, 460, 851, 81, 144, 60, 251, 13, 953, 270, 944, 319, 885, 710,
              952, 517, 278, 656, 919, 377, 550, 207, 660, 984, 447, 553, 338, 234, 383, 749, 916, 626, 
              462, 788, 434, 714, 799, 821, 477, 549, 661, 206, 667, 541, 642, 689, 194, 152, 981, 938, 
              854, 483, 332, 280, 546, 389, 405, 545, 239, 896, 672, 923, 402, 423, 907, 888, 140, 870, 
              559, 756, 25, 211, 158, 723, 635, 302, 702, 453, 218, 164, 829, 247, 775, 191, 732, 115, 
              331, 901, 416, 873, 754, 900, 435, 762, 124, 304, 329, 349, 295, 95, 451, 285, 225, 945, 
              697, 417]
    else:
        raise ValueError(f'Unknown dataset {dataset}.')
    return class_order
