"""
 @Author: ZihaoWang
 @FileName: train_spm.py
 @DateTime: 2024/11/1 20:55
 @SoftWare: PyCharm
 @Function:
"""
import sentencepiece as spm

# 训练 SentencePiece 模型
spm.SentencePieceTrainer.Train(
    input='test.txt',        # 输入的中文语料文件
    model_prefix='chinese_spm',        # 输出模型的前缀，生成 'chinese_spm.model' 和 'chinese_spm.vocab'
    vocab_size=512,                   # 词汇表大小
    character_coverage=0.9995,         # 字符覆盖率，设置为 0.9995 以涵盖大部分汉字
    model_type='bpe'                   # 使用 BPE 模型类型
)