# FasterRCNN-pytorch

FasterRCNN is implemented in VGG, ResNet and FPN base. 

reference:

rbg's FasterRCNN code: [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)

-----
# fasterRCNN 结构介绍
    fasterRCNN是二阶段物体检测框架，是由何大神团队提出，在多个数据集上达到了 最好标准
    二阶段虽然效率低 但是准确率是一阶段无法比肩的
    
    二阶段就是区分一阶段而言的 首先 由RPN网络通过分类和回归 产生建议框 通过特征层 卷积后 产生 2*9 4*9 向量
    其中 分类是 对指定单位进行是否是北京分类得分 回归是计算 框的位置(cx cy w h)
    
    提出了anchor概念；即在输入图片经过主干网络 列如 fpn vgg resnet 卷积生成特征层 然后在特征层上生成9个anchor矩形框
    是有不同比例和size大小生成的9个框 比例（1,2,1/2） size(8 16 32) 然后计算中心点和宽高 加上特征层形成的网络向量
    最后生成的是大量的anchor框  比如800*800 输入图片经过主干网络的特征提取 最后生成50*50的特征层 然后在这个50*50的网格中
    每个网格生成9个anchor矩形框 最终生成的50*50*9个anchor 框 然后去除越界的框 计算bbox针对anchor的偏移 取出得分前12000个（由分类得到的）之后进行
    非极大抑制 去除多余框 保留2000个anchor矩形框
    RPN网络生成建议框的过程
    0.特征提取 进行 分类 回归 2*9 4*9 向量（在resnet上是layer3）
    1. 生成anchor 50*50*9 
    2. 计算bbox针对anchor偏移
    3. 得分过滤和nms过滤得到1200个anchor矩形框
    4. 根据gt计算anchor IOU匹配 anchor和 iou（1.最佳匹配，2.iou>0.7）
    5. 计算gt相对于anchor的偏移为后续过滤 loss计算
    6. labels 正负样本设置 根据iou>0.7 正样本 fg iou<0.3 负样本 bg<0.3
    7. 减少bg fg数量 提高速度，最终128个框参与训练(fg 最大32个 bg 最大128-32个)
    8. 分类计算 和 labels 计算交叉熵 conf_loss
    9. 回归-anchor偏移和gt-anchor偏移进行l1 smooth loss 计算 
    10.产生的建议框和layer4进行卷积第二阶段进行分类和回归
    11.产生的建议框和特征层向量进行roi-pool对齐最为fc层输入
    12.进行fc 分类softmax 以及 回归fc 
    13.计算分类 回归后的loss
    14.根据 rpn loss  和 fasterrcnn loss  进行 bp 反向传播进行更新 参数 不断调整bbox 位置 
    
    
   # 计算bbox和 gt 对anchor 偏移部分
    https://www.cnblogs.com/wangguchangqing/p/10393934.html
    其中心思想就是针对 anchor的偏移于缩放来找到gt最合适的位置 
    其中偏移就是 找到中心点 缩放来 使anchor匹配gt大小 
    1.边境框的回归方法 decode 
    g*(x) = pw*dx + px
    g*(y) = ph*dx + py
    pw*dx 类似于x方向偏移 ph*dx y方向的偏移 
    
    g*(w) = exp(dw)*pw
    g*(h) = exp(dh)*pw
    
    w h 采用线性的函数来调整 因为 指数函数特点 针对 w h 的缩放 
    dx dy dw dh 都是回归学习到的参数 
    
    2.对偏移的计算 encode 于decode 正好相反过程
    

# Model Performance 
### Train on VOC2017 Test on VOC2017  

   | Backbone        | mAp |
   | ----------      |:------:|
   | VGG16 | 0.7061 |
   | ResNet101 | 0.754 |

# Train Your Model
### 1.Before Run You Need:
1. cd ./lib 
 
   > Change gpu_id in make.sh and setup.py.    
   Detially, you need modify parameter setting in line 5, 12 and 19 in make.sh and line 143 in setup.py where include key words '-arch=' depend on your gpu model.(select appropriate architecture described in table below) 
   
   > sh make.sh

    | GPU model        | Architecture    | 
    | --------   | :-----: |
    | TitanX (Maxwell/Pascal)        | sm_52      |
    | GTX 960M        | sm_50 |
    | GTX 108 (Ti)  |sm_61    |
    | Grid K520 (AWS g2.2xlarge)   |sm_30      |
    | Tesla K80 (AWS p2.xlarge)    |sm_37      |

2. cd ../
	 
   > mkdir ./data
	 
   > mkdir ./data/pretrained_model
	 
   > download pre-trained weights in ./data/pretrained_model
   
3. run train.py
   
### 2.How to use?
#### **Note: decentralization in preprocesing is based on BGR channels, so you must guarantee your pre-trained model is trained on the same channel set if you use transfer learning**

For example:

VGG:
CUDA_VISIBLE_DEVICES=1 python train.py --net='vgg16' --tag=vgg16 --iters=70000 --cfg='./experiments/cfgs/vgg16.yml' --weight='./data/pretrained_model/vgg16_caffe.pth'

CUDA_VISIBLE_DEVICES=2 python test.py --net='vgg16' --tag=vgg16 --model=60000 --cfg='./experiments/cfgs/vgg16.yml' --model_path='voc_2007_trainval/vgg16/vgg16_faster_rcnn' --imdb='voc_2007_test' --comp

ResNet:

CUDA_VISIBLE_DEVICES=2 python train.py --net='res18' --tag=res18 --iters=70000 --cfg='./experiments/cfgs/res18.yml' --weight='./data/pretrained_model/Resnet18_imagenet.pth'

CUDA_VISIBLE_DEVICES=3 python train.py --net='res50' --tag=res50 --iters=70000 --cfg='./experiments/cfgs/res50.yml' --weight='./data/pretrained_model/Resnet50_imagenet.pth'

CUDA_VISIBLE_DEVICES=7 python train.py --net='res101' --tag=res101 --iters=80000 --cfg='./experiments/cfgs/res101.yml' --weight='./data/pretrained_model/resnet101_caffe.pth'

CUDA_VISIBLE_DEVICES=6 python test.py --net='res101' --tag=res101_1 --cfg='./experiments/cfgs/res101.yml' --model=70000 --model_path='voc_2007_trainval/res101_1' --imdb='voc_2007_test' --comp

----
