import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, TrainerCallback
from datasets import load_dataset
import datetime
import matplotlib.font_manager as fm


# 指定使用的预训练模型名称
model_name = "bert-base-uncased"  # 替换为你自己的模型路径或名称

# 从预训练模型中加载一个用于序列分类的模型
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 加载对应的分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载 MRPC 数据集
dataset = load_dataset("glue", "mrpc")

# 定义一个预处理函数，用于对数据集中的句子对进行分词和截断
def preprocess_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding=True)

# 对数据集进行预处理，将句子对转换为模型可以接受的输入格式
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 使用 DataCollatorWithPadding 进行动态填充
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 自定义回调类，用于打印每一轮的评估结果
class PrinterCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch}: {kwargs['metrics']}")

# 定义训练参数，包括输出目录等
training_args = TrainingArguments(
    output_dir="test_trainer",  # 输出目录，用于保存模型检查点和其他文件
    per_device_train_batch_size=8,  # 每个设备（如 GPU）上的训练批量大小
    per_device_eval_batch_size=4,  # 每个设备上的评估批量大小
    num_train_epochs=30,  # 训练的轮数
    weight_decay=0.01,  # 权重衰减（L2正则化），用于防止过拟合
    eval_strategy="epoch",  # 评估策略，每个 epoch 结束时进行评估
    logging_dir="logs",  # 日志目录，用于 TensorBoard 等日志记录
    logging_steps=1,  # 日志记录的步数间隔，即每隔多少步记录一次日志
    #fp16=True,  # 启用混合精度训练,有gpu的时候才能使用
)

# 创建一个 Trainer 实例，用于训练和评估模型
trainer = Trainer(
    model=model,  # 模型
    args=training_args,  # 训练参数
    train_dataset=tokenized_datasets["train"],  # 训练数据集
    eval_dataset=tokenized_datasets["validation"],  # 验证数据集
    data_collator=data_collator,  # 动态填充
    callbacks=[PrinterCallback()],  # 添加自定义回调
)

# 进行训练和评估
trainer.train()

# 进行最终评估
eval_result = trainer.evaluate()

# 打印评估结果
print(eval_result)

# 提取指标名称和对应的值
metrics = list(eval_result.keys())
values = list(eval_result.values())

# 创建图表
plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color='skyblue')
plt.xlabel('Metrics指标', fontproperties=font_prop)
plt.ylabel('Values', fontproperties=font_prop)
plt.title('Evaluation Metrics评估指标', fontproperties=font_prop)
plt.ylim(0, 1)  # 设置 y 轴的范围
for i, value in enumerate(values):
    plt.text(i, value + 0.02, f'{value:.2f}', ha='center', fontproperties=font_prop)

# 获取当前日期和时间
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 生成文件名
filename = f"{model_name}_{current_time}.png"

# 保存图表
plt.savefig(filename, bbox_inches='tight')

# 显示图表
plt.show()

print(f"图表已保存为 {filename}")
