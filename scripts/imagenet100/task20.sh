dataset_path=DATASET
python main.py --dataset imagenet100 \
                       --img_size 224 \
                       --task_size 5 \
                       --seed 2021 \
                       --memory_size 2000 \
                       --dataset_path ${dataset_path}