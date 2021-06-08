import json
import torch
from torch.utils.data import Dataset

class SemanticMatchingDataset(Dataset):
    def __init__(self, type):
        self.data = []
        with open("dataset/" + type + ".json") as f:
            for line in f:
                self.data.append(json.loads(line))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, samples):
        n = len(samples)
        scores = torch.zeros((n,))
        sen1, sen2 = [], []
        for i in range(n):
            scores[i] = samples[i]["score"]
            sen1.append(samples[i]["sen1"])
            sen2.append(samples[i]["sen2"])
        return {"scores": scores, "sen1": sen1, "sen2": sen2}

def test_dataset(data_set):
    print("**********")
    print(len(data_set))
    print(data_set[0])
    print(data_set.collate_fn([data_set[0], data_set[1]]))
    print("----------")

if __name__ == "__main__":
    test_dataset(SemanticMatchingDataset("train"))
    test_dataset(SemanticMatchingDataset("valid"))
