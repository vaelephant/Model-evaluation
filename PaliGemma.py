import os
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig
from datasets import load_dataset, DatasetDict

# 定义数据集和模型保存路径
dataset_path = "datasets/VQAv2"
model_id = "google/paligemma-3b-pt-224"
local_model_path = "models/paligemma-3b-pt-224"

# 下载并保存数据集
print("第一步下载保存模型--Downloading and saving dataset...")
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
if not os.path.exists(os.path.join(dataset_path, 'dataset_dict.json')):  # 检查目录是否为空
    print("开始下载数据集：Downloading dataset...")
    ds = load_dataset('HuggingFaceM4/VQAv2', split="train[:10%]")
    ds.save_to_disk(dataset_path)
    print("数据集存入硬盘：Dataset downloaded and saved to disk.")
else:
    print("数据集存在：Loading dataset from disk...")
    ds = DatasetDict.load_from_disk(dataset_path)
    print("数据集从硬盘导入：Dataset loaded from disk.")

# 移除不需要的列
print("Removing unnecessary columns...")
cols_remove = ["question_type", "answers", "answer_type", "image_id", "question_id"]
ds = ds.remove_columns(cols_remove)
print("Unnecessary columns removed.")

# 将数据集进行训练和测试集的划分
print("第二步：切分训练集和测试集--Splitting dataset into training and testing sets...")
split_ds = ds.train_test_split(test_size=0.05)
train_ds = split_ds["train"]
print("Dataset split into training and testing sets.")

# 打印数据集中的一个样本以检查结构
print("打印数据集中的一个样本--Printing a sample from the dataset...")
print(train_ds[0])

# 下载并保存模型到本地
print(" 第三步：下载保存模型 --Downloading and saving model to local disk...")
if not os.path.exists(local_model_path):
    os.makedirs(local_model_path)
if not os.path.exists(os.path.join(local_model_path, 'pytorch_model.bin')):  # 检查目录是否为空
    print("Downloading and saving the model...")
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    processor.save_pretrained(local_model_path)
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model.save_pretrained(local_model_path)
    print("保存模型到硬盘Model downloaded and saved to disk.")
else:
    print("Loading model from disk...")
    processor = PaliGemmaProcessor.from_pretrained(local_model_path)
    model = PaliGemmaForConditionalGeneration.from_pretrained(local_model_path, torch_dtype=torch.bfloat16)
    print("导入模型Model loaded from disk.")

# 设置设备
print("第四步：装载设备-- Setting device to CPU...")
device = "cpu"
model.to(device)

# 冻结模型的部分参数
print("Freezing model parameters...")
for param in model.vision_tower.parameters():
    param.requires_grad = False 
for param in model.multi_modal_projector.parameters():
    param.requires_grad = False 
print("Model parameters frozen.")

# 打印模型模块结构
print("第五步：打印模型-- Model modules:")
for name, module in model.named_modules():
    print(name)

# 根据实际的模块名称调整 target_modules
target_modules = [
    "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
    "vision_tower.vision_model.encoder.layers.0.self_attn.k_proj",
    "vision_tower.vision_model.encoder.layers.0.self_attn.v_proj",
    "vision_tower.vision_model.encoder.layers.0.self_attn.out_proj",
    "vision_tower.vision_model.encoder.layers.1.self_attn.q_proj",
    "vision_tower.vision_model.encoder.layers.1.self_attn.k_proj",
    "vision_tower.vision_model.encoder.layers.1.self_attn.v_proj",
    "vision_tower.vision_model.encoder.layers.1.self_attn.out_proj",
    # 添加更多层...
]

# 调整 LoRA 配置，确保 target_modules 存在于基础模型中
print("第六步：导入lora 配置-- Loading LoRA configuration...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="VQA"
)

try:
    model = get_peft_model(model, lora_config)
    print("LoRA configuration loaded.")
except ValueError as e:
    print(f"Error in LoRA configuration: {e}")

# 打印可训练的参数
print("第七步：打印可训练参数-- Trainable parameters:")
model.print_trainable_parameters()

# 设置训练参数
print("第八步设置训练参数--Setting training arguments...")
args = TrainingArguments(
    num_train_epochs=2,
    remove_unused_columns=False,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=100,
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=1000,
    push_to_hub=True,
    save_total_limit=1,
    output_dir="paligemma_vqav2",
    bf16=True,
    dataloader_pin_memory=False
)
print("Training arguments set.")

# 假设你已经定义了 collate_fn 和 train_ds
# 例如：
def collate_fn(batch):
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'question': [item['question'] for item in batch],
        'answer': [item['answer'] for item in batch]
    }

# 配置 Trainer
print("第九步： 配置训练器--Configuring Trainer...")
trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    data_collator=collate_fn,
    args=args
)
print("Trainer configured.")

# 开始训练
print("第10步训练开始--Starting training...")
trainer.train()
print("第11步训练结束--Training completed.")
