import torch
import aug

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torch.load('model.pth', map_location=device)
model.load_state_dict(torch.load('model_weight.pth', map_location=device))
model.to(device)
model.eval()

def eval_all(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, 
        shuffle=False, num_workers=0)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"total:{total}, correct:{correct}, accuracy:{100*correct/total}%")

def eval(raw_img):
    input = torch.stack((aug.transform(raw_img),)).to(device)
    with torch.no_grad():
        out = model(input)
        print(out)
        _, indices = torch.max(out.data, 1)
        return indices[0].item()

dataset = aug.AugDataset('data', transform=aug.transform)
raw = aug.AugDataset('data')