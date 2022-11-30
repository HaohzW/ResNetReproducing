import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import plain34


BATCH_SIZE = 64
NUM_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = datasets.MNIST(root='data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data',
                              train=False,
                              transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)
valid_loader = test_loader


def compute_accuracy_and_loss(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    cross_entropy = 0.
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        cross_entropy += F.cross_entropy(logits, targets).item()
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100, cross_entropy / num_examples


if __name__ == '__main__':
    model = plain34.Plain34(class_number=10)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_acc_lst, valid_acc_lst = [], []
    train_loss_lst, valid_loss_lst = [], []
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (features, targets) in enumerate(train_loader):
            model.train()
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            logits, probas = model(features)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} | '
                      f'Batch {batch_idx:03d}/{len(train_loader):03d} |'
                      f' loss: {loss:.4f}')
                model.eval()
                with torch.set_grad_enabled(False):
                    train_acc, train_loss = compute_accuracy_and_loss(model, train_loader, device=DEVICE)
                    valid_acc, valid_loss = compute_accuracy_and_loss(model, valid_loader, device=DEVICE)
                    train_acc_lst.append(train_acc)
                    valid_acc_lst.append(valid_acc)
                    train_loss_lst.append(train_loss)
                    valid_loss_lst.append(valid_loss)
                    print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} Train Acc.: {train_acc:.2f}%'
                        f' | Validation Acc.: {valid_acc:.2f}%')
    file = open('testresultforplain34.txt', 'a')
    file.write("train_acc")
    file.write(str(train_acc_lst))
    file.write("valid_acc")
    file.write(str(valid_acc_lst))
    file.write("train_loss")
    file.write(str(train_loss_lst))
    file.write("valid_loss")
    file.write(str(valid_loss_lst))
    file.close()
    torch.save(model, "plain34.pth")
