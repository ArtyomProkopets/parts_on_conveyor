
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


class PartsDataset(Dataset):
    def __init__(self, root, csv_name, transform):
        self.root = Path(root)
        self.rows = pd.read_csv(self.root / csv_name)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        image = Image.open(self.root / row['file']).convert('RGB')
        return self.transform(image), -1


class ResNet18Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 8)

    def forward(self, x):
        return self.model(x)


def predict(data, weights, output):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResNet18Classifier()
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    dataset = PartsDataset(data, 'test.csv', transform)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    predictions = []

    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        'id': dataset.rows.id,
        'label': predictions
    }).to_csv(output, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='.')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', default='submission_reproduced.csv')
    args = parser.parse_args()

    predict(args.data, args.weights, args.output)
