### 对代码的修改
- 使用openai库更现代的api
- 在大模型和智能体类中增加了模型选择
- 增加实验的接口，扩展了 `2d` 和 `scalar` 接口
- 放弃使用 `config/`, 直接从数组中读取 `api_key`, 方式如下

新建 `test/` 文件夹, 并新建`mylib.py`, 导入

```python
from mylib import *
```

模版如下

```python
deepseek_api_keys = [
    "sk-......",
    "sk-......",
]

deepseek_api_base = 'https://api.deepseek.com'

kimi_api_keys = [
    "sk-......",
    "sk-......",
]

kimi_api_base = "https://api.moonshot.cn"

tongyi_api_keys = [
    "sk-......",
    "sk-......",
]

tongyi_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"

__all__ = [
    # 模型信息
    "deepseek_api_base", "deepseek_api_keys", "kimi_api_base", "kimi_api_keys", "tongyi_api_base", "tongyi_api_keys",
]
```

### 实验数据说明

存放在 log/ 文件夹中

`2d_debate/` 和 `scalar_debate/` 测试阶段的数据

后针对3个智能体的情况进行了穷举

生成了`exhaustive/` 和 `exhaustive_random/`

PS: `exhaustive/` 为通义模型。`exhaustive_random/` 为随机模型

