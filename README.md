# yqzwqa
DataFountain 疫情政务问答助手解决方案分享，线下0.755。遗憾的是对应的成绩没有提交成功，不知道线上多少，猜测应该在0.72左右。

## 基本思路

### 搜索
使用ElasticSearch实现，增加对地址的特殊处理。

### 模型
使用Albert的QA模型，采用transformers库的实现。

### 数据
使用搜索结果top 10构造训练集。

### 后处理
使用超参数综合span score，排序分数等因素选出最佳答案。通过简单的文字处理规则优化结果。

## 文件结构
+ train-v8/
    + train-v8.ipynb 训练脚本
    + runs/ tensorboard
+ eval-v8.ipynb 验证集
+ squad-data-v8.ipynb 生成训练数据
+ test-v8.ipynb 生成提交样本
+ addr_dict.p 国内地址库
+ fgm.py FGM对抗样本训练
+ question_answering_pipeline.py 模型推理代码
+ squad.py 模型预处理相关代码
+ squad_metrics.py 模型后处理相关代码
+ user_dict.txt 自定义词典
+ libthulac.so Linux中thulac分词软件预编译c库