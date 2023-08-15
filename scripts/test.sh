export CUDA_VISIBLE_DEVICES=0,1

# kill -9 $(lsof -t /dev/nvidia*)
# sleep 1s
# kill -9 $(lsof -t /dev/nvidia*)
# sleep 1s

config=configs/test_vitbase_jointtraining_config.py

python -m paddle.distributed.launch --log_dir=./logs/vitbase_jointraining --gpus="0,1"  ./ufo_test.py --config-file ${config} --eval-only 


