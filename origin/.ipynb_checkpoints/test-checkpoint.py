import torch
import torch.distributed as dist
from fairscale.nn.model_parallel.layers import ParallelEmbedding
from fairscale.nn.model_parallel import initialize
import os

# 设置环境变量
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'

# 初始化分布式进程组
def init_process_group():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

# 初始化进程组
init_process_group()

# 正确初始化模型并行
initialize.initialize_model_parallel(1)
# 假设有 4 个 GPU，初始化 ParallelEmbedding
vocab_size = 10000  # 示例词汇表大小
embedding_dim = 256  # 示例嵌入维度

# 创建 ParallelEmbedding 实例
parallel_embedding = ParallelEmbedding(vocab_size, embedding_dim)

# 获取当前进程的 rank，并将模型移动到对应的设备
device = torch.device("cuda", dist.get_rank())  # 根据 rank 分配 GPU
parallel_embedding = parallel_embedding.to(device)

# 使用 ParallelEmbedding
input_ids = torch.randint(0, vocab_size, (32, 64)).to(device)
output = parallel_embedding(input_ids)

# 输出结果
print(output)
