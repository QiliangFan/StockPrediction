# 模式识别课程作业 :notebook: -股票价格预测(LSTM和GAN）：

## 1. 运行方式:hammer:：

```bash
# requirements
pip3 install -r pip-requirements.txt
```

```python3
# for lstm
python3 main.py --method lstm

# for gan
python3 main.py --method gan

# for lstm and gan
python3 main.py --method lstm gan
# or
sh run.sh
```


## 2. 配置 :gear:

修改`config.yaml`中的数据集路径，数据集可以从[google drive](https://drive.google.com/drive/folders/1PirLlvWiuZ8posgI1IKL_ljIBQ9HGUkS?usp=sharing)获取。