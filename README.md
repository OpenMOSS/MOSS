# MOSS



## Inference


### blitz
提供不带tool版本的推理脚本，以任何你喜欢的方式load模型之后，  
以任何你喜欢的方式运行：  
`python moss_inference.py`
或者直接在moss_infer_demo.ipynb中探索。

当然由于这是一个不带`Tools`的推理，如果你需要用它来服务，那么你需要至少在别的地方将输入的"<|Commands|>"和"<|Results|>"内的值改为None，并且需要修改部分代码使得for能够在遇到"<eor>"时返回，并且需要一些对于Repetition Penalty的简单修改来跳过Result部分的信息。

### Details

对于显存小于48G，提供了hugginface accelerate的model parallelism方法，该方法需要至少两张3090（24G）。  
采样策略包括: temperature, repetition_penalty, top_k, top_p。




