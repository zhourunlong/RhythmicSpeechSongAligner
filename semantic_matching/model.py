import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class SemanticMatchingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
        self.bert = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
        self.freeze_bert()
        self.lin1 = nn.Linear(768, 768, False)
        self.act = nn.Sigmoid()
        self.lin2 = nn.Linear(1, 1)
        self.act = nn.Sigmoid()        

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = True
    
    def encode(self, sentences):
        tokens = {'input_ids': [], 'attention_mask': []}
        for sentence in sentences:
            new_tokens = self.tokenizer.encode_plus(sentence, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
            tokens['input_ids'].append(new_tokens['input_ids'][0])
            tokens['attention_mask'].append(new_tokens['attention_mask'][0])
        tokens['input_ids'] = torch.stack(tokens['input_ids']).cuda()
        tokens['attention_mask'] = torch.stack(tokens['attention_mask']).cuda()
        embeddings = self.bert(**tokens).last_hidden_state

        mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        return summed / summed_mask

    def forward(self, sen1, sen2):
        embed1 = self.encode(sen1)
        embed2 = self.encode(sen2)

        embed1 = self.lin1(embed1)
        embed2 = self.lin1(embed2)

        feature = nn.CosineSimilarity()(embed1, embed2)
        return 5 * self.act(self.lin2(feature.unsqueeze(-1)).view((-1,)))

if __name__ == "__main__":
    model = SemanticMatchingModel()
    model.cuda()

    sen1 = [
        "Three years later, the coffin was still full of Jello.",
        "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go."
    ]
    sen2 = [
        "The person box was packed with jelly many dozens of months later.",
        "He found a leprechaun in his walnut shell."
    ]

    print(model(sen1, sen2))
