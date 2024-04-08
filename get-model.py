from transformers import AutoTokenizer, BertModel

# Load pre-trained tokenizer and BERT-Mini model
# Save resources locally

tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
model = BertModel.from_pretrained("google/bert_uncased_L-4_H-256_A-4")

tokenizer.save_pretrained("./bert-mini/")
model.save_pretrained("./bert-mini")