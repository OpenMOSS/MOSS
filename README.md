# MOSS
<p align="center" width="100%">
<a href="https://txsun1997.github.io/blogs/moss.html" target="_blank"><img src="https://txsun1997.github.io/images/moss.png" alt="MOSS" style="width: 50%; min-width: 300px; display: block; margin: auto;"></a>
</p>

![](https://img.shields.io/badge/Code_License-Apache_2.0-brightgreen) ![](https://img.shields.io/badge/Data_License-CC_BY--NC_4.0-blue) ![](https://img.shields.io/badge/Model_License-GNU_AGPL_3.0-red)

## 目录

- [开源清单](#开源清单)
  - [模型](#模型)
  - [数据](#数据)
- [介绍](#介绍)
- [本地部署](#本地部署)
- [开源协议](#开源协议)

## 开源清单

### 模型

- [moss-moon-003-base](https://huggingface.co/fnlp/moss-moon-003-base): MOSS-003基座模型，在高质量中英文语料上自监督预训练得到，预训练语料包含约700B单词，计算量约$6.67\times10^{22}$ 次浮点数运算。
- [moss-moon-003-sft](https://huggingface.co/fnlp/moss-moon-003-sft): 基座模型在约110万多轮对话数据上微调得到，具有指令遵循能力、多轮对话能力、规避有害请求能力。
- [moss-moon-003-sft-plugin](https://huggingface.co/fnlp/moss-moon-003-sft): 基座模型在约110万多轮对话数据和约30万插件增强的多轮对话数据上微调得到，在`moss-moon-003-sft`基础上还具备使用搜索引擎、文生图、计算器、解方程等四种插件的能力。
- moss-moon-003-pm: 在基于`moss-moon-003-sft`收集到的偏好反馈数据上训练得到的偏好模型，将在近期开源。
- moss-moon-003: 在`moss-moon-003-sft`基础上经过偏好模型`moss-moon-003-pm`训练得到的最终模型，具备更好的事实性和安全性以及更稳定的回复质量，将在近期开源。
- moss-moon-003-plugin: 在`moss-moon-003-sft-plugin`基础上经过偏好模型`moss-moon-003-pm`训练得到的最终模型，具备更强的意图理解能力和插件使用能力，将在近期开源。

### 数据

- moss-002-sft-data: MOSS-002所使用的多轮对话数据，覆盖有用性、忠实性、无害性三个层面，包含由`text-davinci-003`生成的约50万条英文对话和60万条中文对话。
- moss-003-sft-data: `moss-moon-003-sft`所使用的多轮对话数据，基于MOSS-002内测阶段采集的约10万用户输入数据和`gpt-3.5-turbo`构造而成，相比`moss-002-sft-data`，`moss-003-sft-data`更加符合真实用户意图分布，包含更细粒度的有用性类别标记、更广泛的无害性数据和更长对话轮数，约含110万条对话数据，将在近期开源。
- moss-003-sft-plugin-data: `moss-moon-003-sft-plugin`所使用的插件增强的多轮对话数据，包含支持搜索引擎、文生图、计算器、解方程等四个插件在内的约30万条多轮对话数据，将在近期开源。
- moss-003-pm-data: `moss-moon-003-pm`所使用的偏好数据，包含在约18万额外对话上下文数据及使用`moss-moon-003-sft`所产生的回复数据上构造得到的偏好对比数据，将在近期开源。

## 介绍

MOSS是一个以中英双语为主



## 本地部署



## 开源协议

本项目所含代码采用[Apache 2.0](https://github.com/OpenLMLab/MOSS/blob/main/LICENSE)协议，数据采用[CC BY-NC 4.0](https://github.com/OpenLMLab/MOSS/blob/main/DATA_LICENSE)协议，模型权重采用[GNU AGPL 3.0](https://github.com/OpenLMLab/MOSS/blob/main/MODEL_LICENSE)协议。如需将本项目所含模型用于商业用途或公开部署，请通过robot@fudan.edu.cn联系我们取得授权，商用情况仅用于记录，不会收取任何费用。如使用本项目所含模型及其修改版本提供服务产生误导性或有害性言论，造成不良影响，由服务提供方负责，与本项目无关。

## Inference


### blitz
提供不带tool版本的推理脚本，以任何你喜欢的方式load模型之后，  
以任何你喜欢的方式运行：  
`python moss_inference.py`  
或者直接在moss_infer_demo.ipynb中探索。   

当然由于这是一个不带`Tools`的推理，如果你需要用它来服务，那么你需要至少在别的地方将输入的"<|Commands|>"和"<|Results|>"内的值改为None，并且需要修改部分代码使得for能够在遇到"<eor>"时返回。  

### Details

对于显存小于48G，提供了hugginface accelerate的model parallelism方法，该方法需要至少两张3090（24G）。  



