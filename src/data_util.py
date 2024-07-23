import torch

def encode_data(tokenizer, N, data, batch_size, max_length, device, last_k=None):
    # If the input data is text
    if type(data[0]) == str:
        encodings = tokenizer(data, padding=True, truncation=True, max_length=max_length, return_length=True, return_tensors="pt") # output variable length encodings
        if not last_k:
            encodings = [
                {'input_ids': encodings['input_ids'][i: i + batch_size].to(device),
                'attention_mask': encodings['attention_mask'][i: i + batch_size].to(device),
                'length': encodings['length'][i: i + batch_size] }
                for i in range(0, N, batch_size)
            ]
        else:
            encodings = [
                {'input_ids': encodings['input_ids'][i: i + batch_size][-last_k:].to(device),
                'attention_mask': encodings['attention_mask'][i: i + batch_size][-last_k:].to(device) }
                for i in range(0, N, batch_size)
            ]
    else: # input data is tokens-- manually pad and batch.
        max_len = max([len(sentence) for sentence in data])
        data = [sentence for sentence in data if len(sentence) > 2]
        encodings = [tokenizer.encode(sentence[1:], padding='max_length', max_length=max_len, return_tensors="pt") \
                     for sentence in data]
        batched_encodings = [torch.stack(encodings[i: i + batch_size]).squeeze(1).to(device) for i in range(0, len(data), batch_size)]
        batched_attention_masks = [(tokens != 1).to(device).long() for tokens in batched_encodings]
        encodings = [
            {'input_ids': batched_encodings[j], 'attention_mask': batched_attention_masks[j]}
            for j in range(len(batched_encodings))
        ]

    return encodings