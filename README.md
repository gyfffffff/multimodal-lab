# multimodal-sentiment classification

这是当代人工智能第五次实验的代码仓库。

## 配置

本次作业的实现基于python3.7, 运行代码之前请安装如下依赖:

- torch==1.10.0+cu113

- torchvision==0.11.1+cu113

- transformers==4.30.2

- Pillow==8.4.0

- pandas==1.3.4

- matplotlib==3.5.0
  
- tqdm==4.66.1

- PyYAML==6.0


或你可以直接运行

```python
pip install -r requirements.txt
```

## 代码结构
这里仅标注了主要的文件夹或文件

```python
    multimodal-lab/
    ├─config.yml   # 参数配置文件，在这里修改运行参数
    ├─data/
    │ ├─count_tag.py   # 统计数据集类别占比
    │ ├─data/         # 存放所有数据
    │ │  ├─1.jpg      # 图像模态数据
    │ │  ├─1.txt      # 文本模态数据
    │ │  └─....jpg
    │ ├─test_without_label.txt  # 测试集
    │ ├─train.txt              # 训练集
    │ ├─train_small.txt        # 小训练集（训练集前50条数据）
    │ ├─val.txt               # 验证集
    │ └─val_small.txt           # 小验证集
    ├─logs/                # 存放实验报告中提到的模型训练日志
    ├─res/              # 存放实验报告中提到的模型训练结果折线图
    ├─src/
    │ ├─dataset.py       # 数据预处理
    │ ├─logger.py        # logging包装类，用于记录日志
    │ ├─main.py          # 运行入口
    │ ├─model/           # 存放不同的模型实现
    │ │  ├─bert_resnet_weight.py  # bert和resnet提取文本和图片特征，并加权融合
    │ │  ├─bert_resnet_concat.py  # bert和resnet提取文本和图片特征，并拼接融合
    │ │  └─roberta_swin_att.py   # roberta和swin提取文本和图片特征，并使用注意力机制融合
    │ └─train_val_test.py  # 训练，验证，测试的过程
    └─submit.txt  # 预测结果文件
```


## 运行
1. src/model下实现的任意一个模型都可以运行。例如，你可以通过如下脚本运行bert_resnet_concat:
```python
python src/main.py --modelname bert_resnet_concat --resnet 18
```
你也可以选择一个模态。例如仅使用文本模态：
```python
python src/main.py --modelname bert_resnet_weight --use_image 0
```

2.  你可以选择在小数据集上训练
```python
python src/main.py --modelname bert_resnet_weight --resnet 50 --train_small 1
``` 

注意，所有运行均在CPU上，如果你想在GPU上运行，请在config.yml中将device改为`cuda`


## 模型性能
| Model              | Accuracy |
|--------------------|----------|
| bert               | 0.7202   |
| resnet18           | 0.625    |
| resnet50           | 0.6309    |
| bert+resnet18+concat | 0.7540 |
| bert+resnet50+concat | **0.7617** |
| bert+resnet18+weight | 0.7402 |
| bert+resnet50+weight | 0.7324 |
| roberta+swin+att   | 0.7123   |

## 复现
如要复现以上结果，请运行experiment.sh中的命令。

## 引用

部分代码参考以下仓库：

- [CLMLF](https://github.com/Link-Li/CLMLF)



