from transformers import BertTokenizer, BertForPreTraining, BertForSequenceClassification
import torch

import numpy as np

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertForPreTraining.from_pretrained("bert-base-uncased")

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 4, # question, decision, to-do, None
    output_attentions = False,
    output_hidden_states = False,
)

optimizer = torch.optim.AdamW(model.parameters(), 
                              lr = 5e-5,
                              eps = 1e-08
                              )

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

# prediction_logits = outputs.prediction_logits
# seq_relationship_logits = outputs.seq_relationship_logits

new_sentence = "You should TRAIN this model on a down-stream task." #"Um it 's actually a coyote."

# We need Token IDs and Attention Mask for inference on the new sentence
test_ids = []
test_attention_mask = []

def preprocessing(input_text, tokenizer):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = 32,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt'
                   )

# Apply the tokenizer
encoding = preprocessing(new_sentence, tokenizer)

# Extract IDs and Attention Mask
test_ids.append(encoding['input_ids'])
test_attention_mask.append(encoding['attention_mask'])
test_ids = torch.cat(test_ids, dim = 0)
test_attention_mask = torch.cat(test_attention_mask, dim = 0)

device = 'cpu'

# Forward pass, calculate logit predictions
with torch.no_grad():
  # output = model(test_ids.to(device), token_type_ids = None, attention_mask = test_attention_mask.to(device))
  logits = model(**inputs).logits
print(f'logits : {logits}')
predicted_class_id = logits.argmax().item()
prediction = model.config.id2label[predicted_class_id]
# prediction = 'Spam' if np.argmax(output.logits.cpu().numpy()).flatten().item() == 1 else 'Ham'

print('Input Sentence: ', new_sentence)
print('Predicted Class: ', prediction)