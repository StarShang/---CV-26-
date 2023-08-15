
# export PATH=$PATH:/usr/local/bin
# export PATH=$PATH:/dev/sda2
# export PATH=$PATH:/media/btw/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/include/nccl.h
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/tensorrt/lib
# export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
# export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
# export TENSORRT_DIR=/opt/mmlab/TensorRT-8.4.0.6:$TENSORRT_DIR
# export LD_LIBRARY_PATH=/opt/mmlab/TensorRT-8.4.0.6/lib:$LD_LIBRARY_PATH
# export PATH=/opt/anaconda3/envs/paddle_pz/bin:$PATH

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# kill -9 $(lsof -t /dev/nvidia*)
# sleep 1s
# kill -9 $(lsof -t /dev/nvidia*)
# sleep 1s

config=configs/vitbase_jointtraining_config_copy_batchsize16_epoch200.py

python3 -m paddle.distributed.launch --log_dir=./logs/vitbase_jointraining --gpus="0,1,2,3,4,5,6,7"  ./ufo_train.py --config-file ${config} #--resume 


