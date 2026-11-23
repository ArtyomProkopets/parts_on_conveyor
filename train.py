"""A complete, deliberately simple baseline. Change one thing at a time."""
import argparse, copy, json, time, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

def macro_f1(y_true, y_pred, n_classes=8):
    values = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return float(np.mean(values))

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
        label = int(row['label']) if 'label' in self.rows else -1
        return (self.transform(image), label)



class ResNet18Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            8
        )

    def forward(self, x):
        return self.model(x)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        self.model.layer4.train(mode)
        self.model.fc.train(mode)
        return self


def evaluate(model, loader, device):
    model.eval()
    ys = []
    ps = []
    total_loss = 0
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += nn.functional.cross_entropy(logits, labels).item() * len(labels)
            count += len(labels)
            ys.extend(labels.cpu().tolist())
            ps.extend(logits.argmax(1).cpu().tolist())
    return (total_loss / count, macro_f1(ys, ps), float(np.mean(np.array(ys) == ps)), ys, ps)

def run(data='.', out='runs/baseline', epochs=12, seed=42):
    if epochs < 1:
        raise ValueError('epochs must be positive')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train_ops = [
    transforms.Resize(128),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15)
    ]
    eval_ops = [
        transforms.Resize(128),
    ]
    final_ops = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ]
    train_data = PartsDataset(data, 'train.csv', transforms.Compose(train_ops + final_ops))
    val_data = PartsDataset(data, 'validation.csv', transforms.Compose(eval_ops + final_ops))
    test_data = PartsDataset(data, 'test.csv', transforms.Compose(eval_ops + final_ops))
    generator = torch.Generator().manual_seed(seed)
    tl = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0, generator=generator)
    vl = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=0)
    model = ResNet18Classifier().to(device)

    optimizer = torch.optim.AdamW(
        [
            {
                'params': model.model.layer4.parameters(),
                'lr': 0.0001
            },
            {
                'params': model.model.fc.parameters(),
                'lr': 0.001
            }
        ],
        weight_decay=0.0001
    )
    class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best = -1
    history = []
    start = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for images, labels in tl:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total += len(labels)
        vl_loss, f1, acc, ys, ps = evaluate(model, vl, device)
        history.append({'epoch': epoch + 1, 'train_loss': total_loss / total, 'val_loss': vl_loss, 'val_macro_f1': f1, 'val_accuracy': acc})
        if f1 > best:
            best = f1
            state = copy.deepcopy(model.state_dict())
        print(f'{epoch + 1:02d}: loss {total_loss / total:.3f}; val F1 {f1:.3f}; acc {acc:.3f}', flush=True)
    model.load_state_dict(state)
    torch.save(model.state_dict(), out / 'model.pt')
    pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
    _, f1, acc, ys, ps = evaluate(model, vl, device)
    pd.DataFrame({'id': val_data.rows.id, 'label': ys, 'prediction': ps}).to_csv(out / 'validation_predictions.csv', index=False)
    model.eval()
    pred = []
    with torch.no_grad():
        for images, labels in DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0):
            pred.extend(model(images.to(device)).argmax(1).cpu().tolist())
    pd.DataFrame({'id': test_data.rows.id, 'label': pred}).to_csv(out / 'submission.csv', index=False)
    result = {'seed': seed, 'epochs': epochs, 'best_val_macro_f1': f1, 'val_accuracy': acc, 'parameters': sum((p.numel() for p in model.parameters())), 'seconds': time.time() - start}
    (out / 'results.json').write_text(json.dumps(result, indent=2))
    print(result, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='.')
    p.add_argument('--out', default='runs/baseline')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    run(a.data, a.out, a.epochs, a.seed)