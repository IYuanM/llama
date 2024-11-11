"""
 @Author: ZihaoWang
 @FileName: test_code.py
 @DateTime: 2024/11/10 20:28
 @SoftWare: PyCharm
 @Function:
"""
from tokenizer import Tokenizer
def test_1():
    tokenizer = Tokenizer('chinese_spm.model')
    with open("test.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    print("Input text:", text)
    # 测试 tokenizer.encode() 是否正常工作
    tokenized_text = tokenizer.encode(text,bos=True,eos=True)
    if tokenized_text:
        print("Tokenized output:", tokenized_text)
    else:
        print("Tokenization result is empty. Check the tokenizer or input content.")
def test_2():
    for i in range(0,237 - 512 + 1,512):
        print(i)
if __name__ == '__main__':
    test_2()