"""
 @Author: ZihaoWang
 @FileName: main.py
 @DateTime: 2024/11/1 16:28
 @SoftWare: PyCharm
 @Function:
"""
from transformers import TextDataset,DataCollatorForLanguageModeling,LlamaTokenizer,AutoTokenizer
from torch.utils.data import Dataset,DataLoader
from model import ModelArgs, Transformer
from transformers import AdamW, get_linear_schedule_with_warmup
from tokenizer import Tokenizer
import torch.nn.functional as F
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
# 设置基本配置
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Step 1: 初始化分词器

tokenizer = Tokenizer('llama/origin/chinese_spm.model')
# Step 2: 加载文本数据集
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=512):
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        #logging.info(f"原始文件:{text}")
        # 将文本编码为 token ids
        tokenized_text = tokenizer.encode(text, bos=True,eos=True)
        #logging.info(f"转化tokens:{tokenized_text}")

       # 填充不足的块或者保留原样
        if len(tokenized_text) < block_size:
          self.examples = [tokenized_text]  # 直接保留不足的部分
        else:
          self.examples = [
            tokenized_text[i : i + block_size]
            for i in range(0, len(tokenized_text) - block_size + 1, block_size)
          ]
        if len(tokenized_text) % block_size != 0:
          # 如果最后剩余的部分不足 block_size，保留它
          self.examples.append(tokenized_text[-(len(tokenized_text) % block_size):])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# 创建数据集
dataset = TextDataset("llama/origin/test.txt", tokenizer, block_size=512)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Step 3: 设置数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 对于自回归模型如 LLaMA，不使用 MLM
)

# 模型参数配置
model_args = ModelArgs(
    dim=256,
    n_layers=32,
    n_heads=32,
    vocab_size=tokenizer.n_words,
    max_seq_len=512
)
logging.info("模型参数配置完毕")
# 初始化 LLaMA 模型
model = Transformer(model_args)
model = model.to("cuda")
logging.info("模型初始化完毕")

# 使用 DistributedDataParallel 包装模型
model = DDP(model, device_ids=[torch.cuda.current_device()])
logging.info("模型包装完毕")
# 优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=3e-5)
num_epochs = 3
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=data_collator)
total_steps = len(train_dataloader) * num_epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

for epoch in range(num_epochs):
    model.train()  # 确保模型在训练模式
    logging.info("开始训练")
    logging.info(f"第{epoch+1}个epoch")
    for batch in train_dataloader:
        # 获取输入和标签，并将其移到 GPU
        inputs, labels = batch['input_ids'].to('cuda'), batch['labels'].to('cuda')
        # 获取序列的长度
        seq_len = inputs.size(1)
        logging.info(f"序列长度:{seq_len}")
        # 在每个训练步骤中，start_pos 应该随着 token 逐步递增
        for step in range(seq_len):
            start_pos = step  # 更新 start_pos，每个 token 一步步处理

            # 前向传播
            outputs = model(inputs, start_pos)  # 获取 logits
            logging.info("前向传播完成")
            output_logits = outputs  # 假设模型输出的是 logits
            logging.info(f"模型输出：{output_logits.view(-1, model.module.vocab_size).shape}")
            logging.info(f"标签：{labels}")
            logging.info(f"修改后的：{labels[:, 2].view(-1)}")

            # 目标是下一个 token，labels 是目标 token
            loss = F.cross_entropy(output_logits.view(-1, model.module.vocab_size), labels[:, step].view(-1))
            logging.info("损失计算完成")
            # 反向传播
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新模型参数
            scheduler.step()  # 更新学习率（如果使用学习率调度器）
            logging.info("反向传播完成")
            # 打印当前的损失
            print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.item()}")