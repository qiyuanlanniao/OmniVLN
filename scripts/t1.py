# llava_test.py
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

# 模型路径
model_path = "/home/iot/hm/ros2_ws/models/llava-v1.6-mistral-7b"

# 图片路径
image_path = "/home/iot/Pictures/Screenshots/Screenshot from 2026-02-20 19-23-02.png"

# 1️⃣ 加载模型和处理器
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# 2️⃣ 加载图片
image = Image.open(image_path).convert("RGB")

# 3️⃣ 准备输入（假设我们想让模型描述图片）
prompt = "Describe the content of this image."

inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

# 4️⃣ 模型生成输出
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=200)
    
# 5️⃣ 解码生成文本
generated_text = processor.decode(output_ids[0], skip_special_tokens=True)
print("Model Output:\n", generated_text)
