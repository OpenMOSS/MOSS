# MOSS



## Inference


### blitz
提供不带tool版本的推理脚本，以任何你喜欢的方式load模型之后，  
以任何你喜欢的方式运行：  
`python moss_inference.py`
或者直接在moss_infer_demo.ipynb中探索。

### Details

对于显存小于48G，提供了hugginface accelerate的model parallelism方法，该方法需要至少两张3090（24G）。  
采样策略包括: temperature, repetition_penalty, top_k, top_p。




