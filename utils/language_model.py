from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import os

# ── 语言 → 模型映射 ───────────────────────────────────────────────────────────
_QWEN_LANGS = {'中文', '英文', '日文'}   # Qwen2.5 支持的语言

def get_llm_model_name(language: str, llm_model: str = None) -> str:
    """根据语言返回应使用的 LLM 模型名称。
    中文/英文/日文 → Qwen2.5-32B-Instruct；其余欧洲语言 → gemma-3-27b-it。
    """
    if llm_model:
        return llm_model
    return 'Qwen2.5-32B-Instruct' if language in _QWEN_LANGS else 'gemma-3-27b-it'


def load_model(model_name="Qwen2.5-32B-Instruct"):
    _LLM_DIR = os.environ.get("LLM_MODEL_DIR", "./LLMs")
    model_path = os.path.join(_LLM_DIR, model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def generate_response(
    model, tokenizer,
    sys_prompt = "You are a helpful assistant.",
    prompt = "你是什么样的模型？",
    max_new_tokens=2048,
    temperature=0.01,
    do_sample=None,
    repetition_penalty=1.2,
    enable_thinking=False,
    ):


    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # temperature > 0 时必须 do_sample=True，否则模型会忽略 temperature
    if do_sample is None:
        do_sample = temperature > 0
    elif not do_sample:
        temperature = 0

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, 
    generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

if __name__ == "__main__":
    # gemma-3-27b-it / Qwen2.5-32B-Instruct / gpt-oss-20b
    
    language = "中文"
    model, tokenizer = load_model('Qwen3.5-9B')
    prompt = "用微信聊天的非正式的，简短的语气回答我所有的消息。你的角色是我的女朋友"

    response = generate_response(model, tokenizer, prompt=prompt)
    print("模型回复：", response)
    