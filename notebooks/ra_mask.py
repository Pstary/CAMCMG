
from transformers import AutoTokenizer

def test_codet5_tokenizer():
    # 加载 CodeT5 fast tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base", use_fast=True)
    
    print("Is fast tokenizer:", tokenizer.is_fast)  # 应该是 True

    # 准备示例代码文本
    example_text = "def add(a, b): return a + b"
    
    # 编码，返回 batch encoding
    encoding = tokenizer(example_text, return_offsets_mapping=True, return_tensors=None)
    
    # 调用 word_ids
    word_ids = encoding.word_ids(batch_index=0)
    
    print("word_ids:", word_ids)
    print("Tokens:", tokenizer.convert_ids_to_tokens(encoding["input_ids"]))
    
if __name__ == "__main__":
    test_codet5_tokenizer()
