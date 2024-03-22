# ---CV-26-



第二届广州·琶洲算法大赛-智能交通CV模型赛题第8名方案
=====

参考自：https://github.com/Traffic-X/Open-TransMind/tree/main/PAZHOU/base

### 具体请参考说明：原理说明.doc  和 运行说明.doc

模型说明：
-------
VIMER-UFO（UFO：Unified Feature Optimization） All in One 多任务训练方案，通过使用多个任务的数据训练一个功能强大的通用模型，可被直接应用于处理多个任务。不仅通过跨任务的信息提升了单个任务的效果，并且免去了下游任务 fine-tuning 过程。VIMER-UFO All in One 研发模式可被广泛应用于各类多任务 AI 系统，以智慧城市场景为例，VIMER-UFO 可以用单模型实现人脸识别、人体和车辆ReID等多个任务的 SOTA 效果，同时多任务模型可获得显著优于单任务模型的效果，证明了多任务之间信息借鉴机制的有效性。

![image](https://github.com/StarShang/---CV-26-/assets/51013149/c6037433-8c09-49ee-b0a5-b17ca091151e)



图片来自论文：2207.10341v1.pdf (arxiv.org)





应用说明：
----------



![image](https://github.com/StarShang/---CV-26-/assets/51013149/ca569a0b-0858-4307-ab11-ef887beafe38)









优化提升项：
---------
#### 一、数据增强：

（1）去除语义分割数据导入 的 随机内容变换 ，  保留 resize和 随机填充裁剪



（2）去除分类数据导入时的自动增强


（3）去除目标检测数据导入 RandomShortSideResize和crop



#### 二、增强网络的正则化表达：

（1）目标检测网络Dropout比例设为0.3 解决co-adaption问题，这样可以训练更宽的网络
 
#### 三、学习率优化：

  （1）学习率优化schedule 改为  PolynomialDecay，初始学习率改为 base_lr 设置为1e-5   （策略：小步慢走，越到后面 步子要迈的越小）

#### 四、梯度下降策略：

（1）实现梯度累加策略 ，相当于增加batchsize，不是每次迭代都进行参数更新，再进行梯度清零，而是若干次迭代后再进行此操作。这样可以增加训练速度。


数据配置
--------

从官方数据下载地址 https://aistudio.baidu.com/datasetdetail/211902 下载训练和测试数据后，将数据解压到datasets文件夹中（若不存在，请先创建）

训练
-----

使用object365数据集的预训练权重 https://bj.bcebos.com/v1/ai-studio-online/2fbaaaaf6bcc424393504d673ef67cfaeb252de248194cc49e3bd6544c2d0e0c?responseContentDisposition=attachment%3B%20filename%3Ddino_vit-base_paddle.pdmodel&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2023-06-04T02%3A29%3A31Z%2F-1%2F%2Fc3f75ecc5bb19e454ec3b1128c6c8ab55f76863be1134813247fe38211562597 ，下载预训练权重至pretrained文件夹中（若不存在，请先创建），后使用以下脚本在训练集上启动训练

#### 双卡3090服务器运行说明：


（1）参考 《环境说明.txt》 安装虚拟conda环境 命名 paddle_conda_env

（2）执行命令：

        conda activate paddle_conda_env

        sh scripts/train.sh  
      
![image](https://github.com/StarShang/---CV-26-/assets/51013149/f6d7dcf0-900c-4e9d-ad84-bbcce10bcfc8)

      
#### 8卡A100 服务器运行说明：   

（1）参考 《环境说明.txt》 安装虚拟conda环境 命名 paddle_conda_env

（2）执行命令：

         conda activate paddle_conda_env
         
         sh scripts/train_device_8_batchsize_16_epoch_200.sh  


预测
------

训练完成后 model_final.pdmodel 模型保存在outputs\vitbase_joint_training 文件夹中（若不存在，请先创建），后使用以下脚本在测试集上启动预测

      sh scripts/test.sh

预测日志：

[08/05 09:30:40 ufo]: Full config saved to outputs/test_vitbase_joint_training/config.yaml
missing keys: []
unexpected keys: []
trans: [<data.transforms.seg_transforms.Normalize object at 0x7f6d4e558650>] <class 'omegaconf.listconfig.ListConfig'>
InferDataset has 1000 samples
InferDataset has 1000 samples after padding
rank 1 has 500 items
loading annotations into memory...
Done (t=0.01s)
creating index...
index created!
COCOInferDataSet has 3067 samples
COCOInferDataSet has 3068 samples after padding
rank 1 has 1534 items
========== InferDataset ==========
seg_inference_on_test_dataset

  0%|          | 0/500 [00:00<?, ?it/s]
  8%|▊         | 38/500 [00:10<02:02,  3.79it/s]
  8%|▊         | 38/500 [00:20<02:02,  3.79it/s]
 18%|█▊        | 89/500 [00:20<01:30,  4.52it/s]
 18%|█▊        | 89/500 [00:30<01:30,  4.52it/s]
 28%|██▊       | 139/500 [00:30<01:16,  4.72it/s]
 38%|███▊      | 190/500 [00:40<01:04,  4.84it/s]
 48%|████▊     | 241/500 [00:50<00:53,  4.88it/s]
 58%|█████▊    | 291/500 [01:00<00:42,  4.91it/s]
 69%|██████▊   | 343/500 [01:10<00:31,  4.99it/s]
 79%|███████▉  | 395/500 [01:21<00:21,  4.98it/s]
 89%|████████▉ | 445/500 [01:31<00:11,  4.95it/s]
 99%|█████████▉| 497/500 [01:41<00:00,  5.02it/s]
100%|██████████| 500/500 [01:42<00:00,  4.89it/s]
========== FGVCInferDataset ==========
========== COCOInferDataSet ==========
['/home/shangzaixing/code/PAZHOUbase', '/opt/anaconda3/envs/paddle_pz/lib/python37.zip', '/opt/anaconda3/envs/paddle_pz/lib/python3.7', '/opt/anaconda3/envs/paddle_pz/lib/python3.7/lib-dynload', '/opt/anaconda3/envs/paddle_pz/lib/python3.7/site-packages', '/opt/anaconda3/envs/paddle_pz/lib/python3.7/site-packages/paddle/fluid/proto', '/home/shangzaixing/code/PAZHOUbase/']
['/home/shangzaixing/code/PAZHOUbase', '/opt/anaconda3/envs/paddle_pz/lib/python37.zip', '/opt/anaconda3/envs/paddle_pz/lib/python3.7', '/opt/anaconda3/envs/paddle_pz/lib/python3.7/lib-dynload', '/opt/anaconda3/envs/paddle_pz/lib/python3.7/site-packages', '/opt/anaconda3/envs/paddle_pz/lib/python3.7/site-packages/paddle/fluid/proto', '/home/shangzaixing/code/PAZHOUbase/']
/home/shangzaixing/code/PAZHOUbase/detectron2/data/transforms/transform.py:46: DeprecationWarning: LINEAR is deprecated and will be removed in Pillow 10 (2023-07-01). Use BILINEAR or Resampling.BILINEAR instead.
  def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):
rank is 1, world size is 2




