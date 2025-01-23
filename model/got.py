from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = ""

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda:"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# history
question = "what did the baby do after he turned around near the end of the video？"
prompt = "There is a little boy, a young child and his mom. The little boy is playing with a toy and wearing a tie. The young child is also playing with a toy. And, the mom is taking care of the little boy and young child."
fact = "a) hug boy	b) flail around	c) touch the camera	d) push box to where he started	e) fall"
messages = [
    {"role": "system", "content": "你是一个内容理解与推理专家，你需要做的是从我的场景描述中进行内容理解和推理，请基于问题，推断出场景，并回答我的问题。"},
    {"role": "assistant", "content": "请回答我的问题：The question is"+question+". The options is "+fact+". 回答格式：【选项k)】"},
    {"role": "user", "content": "请根据我给你的内容一步一步考虑：1) 构建scene graph 2) 对每个选项进行分析，对每个选项打一个1-10的评分。3）给出最终的答案。The scenario is："+prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print('step 3: score')
print(response)