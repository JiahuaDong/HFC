dataset_path=DATASET
python main.py --dataset cifar100 \
                       --img_size 224 \
                       --task_size 2 \
                       --init_classes 2 \
                       --seed 2021 \
                       --model_depth 11 \
                       --memory_size 2000 \
                       --dataset_path ${dataset_path}