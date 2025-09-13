"""
MIT License

Copyright (c) [2023] [Intelligent Unmanned Systems Laboratory at 
Westlake University]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE, OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import openai
import math
import yaml


# 获取当前脚本的绝对路径
import os
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

# Load the configuration from the YAML file(更改为绝对路径导入)
with open(f'{project_root}/config/keys.yml', 'r') as config_file:
    # 读取为字典
    config = yaml.safe_load(config_file)
openai.api_base = config.get('api_base', '')
api_keys_all = config.get('api_keys', {})
# User ID for which we need to slice the dictionary.
user_id = 2
# Total number of users among whom the dictionary needs to be distributed.
user_count = 3
# Calculate the number of keys each user should get.
keys_per_user = math.floor(len(api_keys_all) / user_count)# 向上取整

# Calculate the starting and ending index for slicing the dictionary 
# for the given user_id.
'''
eg.
    user_count = 3
    api_keys_all = 28
    keys_per_user = math.ceil(28 / 3) = 10
        如果`user_id 是 0`
            start = 10 * 0 = 0
            end = min(10, 28) = 10
            获得 0- 9的api-key
        如果`user_id 是 1`
            start = 10 * 1 = 10
            end = min(20, 28) = 20
            获得10-19的api-key
        如果`usr_id 是 2`
            start = 10 * 2 = 20
            end = min(30, 28) = 28
            获得20-27的api-key
'''
start = keys_per_user * user_id
end = min(keys_per_user * (user_id + 1), len(api_keys_all))
print("user {}/{} ,api_key index start: {}, end: {}"
      .format(user_id, user_count, start, end))
# Slicing the dictionary based on the calculated start and end.
api_keys = {i - start: v 
            for i, (k, v) in enumerate(api_keys_all.items()) 
            if start <= i < end}

if __name__ == '__main__':
    print(api_keys)
