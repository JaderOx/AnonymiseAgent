import os
from openai import OpenAI

def load_openai():
    client = OpenAI(
        base_url=os.environ.get("BASE_URL"),
        api_key=os.environ.get("API_KEY"),
    )
    return client

def generate_response(model="gpt-5-nano-2025-08-07", client=None, sys_prompt="You are a helpful assistant.",
                      prompt=None, temperature=0.01, max_tokens=2048, **kwargs):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        # temperature=temperature,
        # max_tokens=max_tokens,
    )
    return completion.choices[0].message.content

def recognize(model_result):
    return model_result.content

if __name__ == "__main__":
    model = "gpt-5-nano-2025-08-07"
    prompt = "不要多余输出任何东西，请输出<result>None</result>"
    client = load_openai()
    res = generate_response(model=model, client=client, prompt=prompt)
    print(res)
