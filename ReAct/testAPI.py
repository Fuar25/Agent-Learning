from openai import OpenAI

aliyun_api_key = 'sk-5f9decba5c60449aa66631604d8aa4cf'
client = OpenAI(
    api_key=aliyun_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = client.chat.completions.create(
    model="MiniMax-M2.1",
    messages=[
        {'role': 'user', 'content': "你是谁？"}
    ]
)

# 打印完整回答内容
print(response.choices[0].message.content)