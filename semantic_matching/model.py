import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class SemanticMatchingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
        self.bert = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
        self.freeze_bert()
        self.lin = nn.Linear(768, 768, False)
        self.act = nn.Sequential(
            nn.Linear(1, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

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

        embed1 = self.lin(embed1)
        embed2 = self.lin(embed2)

        feature = nn.CosineSimilarity()(embed1, embed2).unsqueeze(-1)
        return 5 * self.act(feature).view(-1,)
    
    def calc_similarity(self, sen1, sen2):
        embed1 = self.encode(sen1)
        embed2 = self.encode(sen2)

        embed1 = self.lin(embed1)
        embed2 = self.lin(embed2)

        # manually calculate cosine similarity
        similarity = torch.matmul(embed1, embed2.transpose(0, 1))
        norm1 = torch.norm(embed1, dim=1, keepdim=True)
        norm2 = torch.norm(embed2, dim=1, keepdim=True)
        norm = torch.matmul(norm1, norm2.transpose(0, 1))
        norm[norm < 1e-8] = 1e-8
        similarity /= norm

        return 5 * self.act(similarity.view((-1, 1))).view(norm1.shape[0], -1)

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
    sen3 = [
        "The person box was packed with jelly many dozens of months later.",
        "He found a leprechaun in his walnut shell.",
        "Though one must not pick flowers on the way up, the blossoms trailed in as they passed."
    ]

    print(model(sen1, sen2))

    print(model.calc_similarity(sen1, sen3))
