# MOSS API接口文档

## 1 REST API

**请求URL：**http://ip_address/api/inference

**请求方法：**POST

**字符编码格式：**UTF-8

**请求头**：

Header 中添加 apikey: APIKEY，由接口方提供

**请求参数：**

放在 HTTP Body 中，采用 json 格式，具体参数如下：

| 请求参数名 | 类型   | 参数说明          | 是否必传 | 备注                                |
| ---------- | ------ | ----------------- | -------- | ----------------------------------- |
| context    | string | 对话的上下文信息  | 否       | 必须和上次请求返回的context保持一致 |
| request    | string | 当前的对话请求    | 是       |                                     |
| plugin     | object | 开启/关闭插件功能 | 否       | 暂未实现                            |

**返回结果：**

| **参数名称** | 类型   | 参数说明                     | 是否必返 | 备注                       |
| ------------ | ------ | ---------------------------- | -------- | -------------------------- |
| response     | string | MOSS 回复的信息              | 是       |                            |
| context      | string | 这次回复生成的所有上下文信息 | 是       |                            |
| extra_data   | array  | 使用 Plugin 后生成的额外信息 | 否       | 建议先不解析，随时可能变动 |

其中 extra_data 是一个 array，其中每个 object 的结构为

```json
{
    "type": "xxx",
    "request": "xxx",
    "data": "xxx"
}
```

其中 `data` 是一次 plugin (tools) 请求返回的原始数据。

### 示例

请求示例：不含有上下文

```json
{
    "request": "hi"
}
```

返回示例：

```json
{
    "response": "Hello! How may I assist you today?",
    "context": "<|Human|>: hi<eoh>\n<|Inner Thoughts|>: None<eot>\n<|Commands|>: None<eoc>\n<|Results|>: None<eor>\n<|MOSS|>: Hello! How may I assist you today?<eom>",
    "extra_data": null
}
```

或者是

```json
{
    "response": "Hello! How may I assist you today?",
    "context": "<|Human|>: hi<eoh>\n<|Inner Thoughts|>: None<eot>\n<|Commands|>: None<eoc>\n<|Results|>: None<eor>\n<|MOSS|>: Hello! How may I assist you today?<eom>"
}
```

请求示例：含有上下文

```json
{
    "context": "<|Human|>: hi<eoh>\n<|Inner Thoughts|>: None<eot>\n<|Commands|>: None<eoc>\n<|Results|>: None<eor>\n<|MOSS|>: Hello! How may I assist you today?<eom>",
    "request": "what's your name?"
}
```

返回示例：

```json
{
    "response": "My name is Moss. How about we get started with some basic questions so let me know how it goes for both of us?",
    "context": "<|Human|>: hi<eoh>\n<|Inner Thoughts|>: None<eot>\n<|Commands|>: None<eoc>\n<|Results|>: None<eor>\n<|MOSS|>: Hello! How may I assist you today?<eom><|Human|>: what's your name?<eoh>\n<|Inner Thoughts|>: 0  \n<|Commands|>: \"name\"  \n<|Results|>: None  \n<|MOSS|>: My name is Moss. How about we get started with some basic questions so let me know how it goes for both of us?<eom>",
    "extra_data": null
}
```

或者是

```json
{
    "response": "My name is Moss. How about we get started with some basic questions so let me know how it goes for both of us?",
    "context": "<|Human|>: hi<eoh>\n<|Inner Thoughts|>: None<eot>\n<|Commands|>: None<eoc>\n<|Results|>: None<eor>\n<|MOSS|>: Hello! How may I assist you today?<eom><|Human|>: what's your name?<eoh>\n<|Inner Thoughts|>: 0  \n<|Commands|>: \"name\"  \n<|Results|>: None  \n<|MOSS|>: My name is Moss. How about we get started with some basic questions so let me know how it goes for both of us?<eom>"
}
```

## http错误码

500：发送的JSON格式错误或者infer服务器连接错误和故障，返回

```json
{
    "code": 500,
    "message": "xxx"
}
```

400：发送的 request 为空或者没有 request 字段，返回

```json
{
    "code": 400,
    "message": "Validation Error: invalid request\n",
    "detail": [
        {
            "FieldError": {},
            "field": "request",
            "tag": "min",
            "value": "1"
        }
    ]
}
```

400：到达 infer 长度最长限制，返回

```json
{
    "code": 400,
    "message": "The maximum context length is exceeded",
    "message_type": "max_length"
}
```

400：检测到输入输出任意敏感信息，返回

```json
{
    "code": 400,
    "message": "Sorry, I have nothing to say. Try another topic. I will block your account if we continue this topic :)",
    "message_type": "sensitive"
}
```

