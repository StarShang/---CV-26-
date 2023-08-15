# ---CV-26-
参考：https://github.com/Traffic-X/Open-TransMind/tree/main/PAZHOU/base
具体请参考说明：原理说明.doc  和 运行说明.doc
=====

第二届广州·琶洲算法大赛-智能交通CV模型赛题第26名方案
=======
模型说明：
-------
VIMER-UFO（UFO：Unified Feature Optimization） All in One 多任务训练方案，通过使用多个任务的数据训练一个功能强大的通用模型，可被直接应用于处理多个任务。不仅通过跨任务的信息提升了单个任务的效果，并且免去了下游任务 fine-tuning 过程。VIMER-UFO All in One 研发模式可被广泛应用于各类多任务 AI 系统，以智慧城市场景为例，VIMER-UFO 可以用单模型实现人脸识别、人体和车辆ReID等多个任务的 SOTA 效果，同时多任务模型可获得显著优于单任务模型的效果，证明了多任务之间信息借鉴机制的有效性。


图片来自论文：2207.10341v1.pdf (arxiv.org)





应用说明：
----------












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
#### 8卡A100 服务器运行说明：   

（1）参考 《环境说明.txt》 安装虚拟conda环境 命名 paddle_conda_env

（2）执行命令：
conda activate paddle_conda_env
      sh scripts/train_device_8_batchsize_16_epoch_200.sh  


预测
------

我们提供了我们训练的三任务AllinOne联合训练的权重，可下载权重至pretrained文件夹中（若不存在，请先创建），后使用以下脚本在测试集上启动预测

sh scripts/test.sh




