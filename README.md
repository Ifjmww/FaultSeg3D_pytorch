# FaultSeg3D_pytorch
FaultSeg3D的pytorch版本(个人复现结果，如与FaultSeg3D原版代码有出入，以原版代码为主)

### [原文链接](http://cig.ustc.edu.cn/_upload/tpl/05/cd/1485/template1485/papers/wu2019FaultSeg3D.pdf)

![FaultSeg3D网络结构图](/docs/FaultSeg3D.png "FaultSeg3D")

## 运行
### 配置环境
* [requirements.txt](./requirements.txt)
#### Train(默认参数设置与文中相同)
```angular2html
python main.py --mode train --exp [experiment_name] --train_path [train_dataset_path] --valid_path [valid_dataset_path]
```
#### Valid_Only(需要有预训练模型)
```angular2html
python main.py --mode valid_only --exp [experiment_name] --valid_path [valid_dataset_path]
```
#### Prediction(需要有预训练模型)
```angular2html
python main.py --mode pred --exp [experiment_name] --pretrained_model_name [FaultSeg3D_BEST.pth] --pred_data_name [pretrained_model_name] 
```

### 训练集、验证集、预测集
* 上述数据均已做预处理: (1) dat->npy; (2) 正则化(减均值除标准差);
* 训练集及验证集-200个数据--[百度网盘链接](https://pan.baidu.com/s/10o848E2vMmjmi21xZBFRiw?pwd=i4mo)-提取码:i4mo 
* 训练集及验证集-800个数据(数据增强)--[百度网盘链接](https://pan.baidu.com/s/1PzsmRt9drnZI9J5GFOk9rw?pwd=zwqf)-提取码:zwqf 
* 预测集-f3数据--[百度网盘链接](https://pan.baidu.com/s/1iBnW94Yn2U0GQQF3-3pXOA?pwd=0b2j)-提取码:0b2j


## 实验结果
### 合成地震数据断层分割结果
![合成地震数据断层分割结果](/docs/合成数据结果.png "合成地震数据断层分割结果")
### 荷兰F3真实地震数据断层分割结果
![荷兰F3真实地震数据断层分割结果](/docs/F3结果.png "荷兰F3真实地震数据断层分割结果")

## 归属声明 / Attribution Statement :

如果您在您的项目中使用或参考了本项目（FaultSeg3D_pytorch）的代码，我们要求并感激您在项目文档或代码中包含以下归属声明：
```commandline
本项目使用了Ifjmww在GitHub上的FaultSeg3D_pytorch项目的代码，特此致谢。原项目链接：https://github.com/Ifjmww/FaultSeg3D_pytorch
```
我们欣赏并鼓励开源社区成员之间的相互尊重和学习，感谢您的合作与支持。

&nbsp;

If you use or reference the code from this project (FaultSeg3D_pytorch) in your project, we require and appreciate an attribution statement in your project documentation or code as follows:
```commandline
Parts of this code are based on modifications of Ifjmww's FaultSeg3D_pytorch. Original project link: https://github.com/Ifjmww/FaultSeg3D_pytorch
```
We value and encourage mutual respect and learning among members of the open source community. Thank you for your cooperation and support.