import sys
# add torchvision repos to import path
sys.path.append("/Users/skaor/Documents/master/idp/pytorch/vision/references/classification")
import helper
from aug import dataset
import torch
import torch.nn as nn
import torch.optim as optim
from train import train_one_epoch, evaluate

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = dataset.num_class()

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
num_epochs = 40

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

def main():
    # Initialize the model for this run
    # input_size of ResNet50 should be 224
    model, input_size = helper.initialize_model(
        model_name, num_classes, feature_extract, use_pretrained=True)
    # Print the model we just instantiated
    print(model)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    # randomize indix in order to remove class bias
    indices = torch.randperm(len(dataset)).tolist()
    # only 20% of the data used for evaluation
    test_size = int(len(dataset) * 0.2)
    dataset_train = torch.utils.data.Subset(dataset, indices[:-test_size])
    dataset_val = torch.utils.data.Subset(dataset, indices[-test_size:])
    dataloaders = {
        'train': torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=0),
        'val': torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)
    }
    # Send the model to GPU
    model = model.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss().to(device)

    # Train and evaluate
    # model, hist = utils.train_model(model, dataloaders, criterion, optimizer, device, num_epochs=num_epochs, is_inception=(model_name=="inception"))
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, criterion, optimizer, dataloaders['train'],
                        device, epoch, print_freq=10)
        # evaluate on the test dataset
        evaluate(model, criterion, dataloaders['val'], device=device)

    print("All done")
    torch.save(model, 'model.pth')
    torch.save(model.state_dict(), 'model_weight.pth')

if __name__ == '__main__':
    main()