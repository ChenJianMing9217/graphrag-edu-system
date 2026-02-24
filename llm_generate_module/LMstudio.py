from openai import OpenAI

client = OpenAI(
    base_url="http://10.242.108.11:8000/v1",
    api_key="lm-studio"  # LM Studio 不強制驗證，填什麼都行
)

resp = client.chat.completions.create(
    model="qwen2.5-7b-instruct",
    messages=[{"role":"user","content":"Hello"}],
    temperature=0.2,
)
print(resp.choices[0].message.content)
