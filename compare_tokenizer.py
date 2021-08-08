from transformers import ReformerTokenizer as PTReformerTokenizer

from reformer import ReformerTokenizer as PDReformerTokenizer

pt_tokenizer = PTReformerTokenizer.from_pretrained(
    "google/reformer-crime-and-punishment"
)
pd_tokenizer = PDReformerTokenizer.from_pretrained("reformer-crime-and-punishment")


text = "It is a nice day today , I want to go to the park !"


o1 = pt_tokenizer.tokenize(text)
o2 = pt_tokenizer.tokenize(text)

print(o1)
print(o2)

print("=" * 50)

o1 = pt_tokenizer(text)
o2 = pt_tokenizer(text)

print(o1)
print(o2)

"""
['▁I', 't', '▁is', '▁a', '▁n', 'i', 'ce', '▁d', 'ay', '▁to', 'd', 'ay', '▁', ',', '▁I', '▁w', 'ant', '▁to', '▁go', '▁to', '▁the', '▁p', 'ar', 'k', '▁', '!']
['▁I', 't', '▁is', '▁a', '▁n', 'i', 'ce', '▁d', 'ay', '▁to', 'd', 'ay', '▁', ',', '▁I', '▁w', 'ant', '▁to', '▁go', '▁to', '▁the', '▁p', 'ar', 'k', '▁', '!']
==================================================
{'input_ids': [33, 260, 111, 4, 136, 264, 69, 30, 71, 26, 268, 71, 258, 277, 33, 8, 180, 26, 224, 26, 13, 40, 52, 282, 258, 287], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
{'input_ids': [33, 260, 111, 4, 136, 264, 69, 30, 71, 26, 268, 71, 258, 277, 33, 8, 180, 26, 224, 26, 13, 40, 52, 282, 258, 287], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
"""
