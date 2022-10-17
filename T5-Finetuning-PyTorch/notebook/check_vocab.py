from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('./kt-ulm-base/')
vocab = tokenizer.get_vocab()
for i in ['PS','LC','TI','QT','DT','OG']:
    print(i,':', i in vocab)