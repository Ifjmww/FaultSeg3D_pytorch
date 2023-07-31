# FaultSeg3D_pytorch
FaultSeg3D的pytorch版本

### [原文链接](http://cig.ustc.edu.cn/_upload/tpl/05/cd/1485/template1485/papers/wu2019FaultSeg3D.pdf)

![FaultSeg3D网络结构图](/docs/FaultSeg3D.png "FaultSeg3D")

### 运行
#### 配置环境
* [requirements.txt](./requirements.txt)
#### Train(默认参数设置与文中相同)
```angular2html
python main.py --mode train --exp [XXX] --train_path [XXX] --valid_path [XXX]
```
#### Valid_Only(需要有预训练模型)
```angular2html
python main.py --mode valid_only --exp [XXX] --valid_path [XXX] 
```
#### Prediction(需要有预训练模型)
```angular2html
python main.py --mode pred --exp [XXX] --pretrained_model_name [XXX.pth] --pred_data_name [XXX] 
```

### 训练集、验证集、预测集
* 上述数据均已做预处理: (1) dat->npy; (2) 正则化(减均值除标准差);
* 训练集及验证集-200个数据--[百度网盘链接](https://pan.baidu.com/s/10o848E2vMmjmi21xZBFRiw?pwd=i4mo)-提取码:i4mo 
* 训练集及验证集-800个数据(数据增强)--[百度网盘链接](https://pan.baidu.com/s/1PzsmRt9drnZI9J5GFOk9rw?pwd=zwqf)-提取码:zwqf 
* 预测集-f3数据--[百度网盘链接](https://pan.baidu.com/s/1iBnW94Yn2U0GQQF3-3pXOA?pwd=0b2j)-提取码:0b2j
