import torch
from sharpen import sharpen
from tile_adjacent import tile_adjacent
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def guess_targets(features, model, output_transform, K, T):

    # original_device = features.device
    with torch.no_grad():
#        features = features.to(next(model.parameters()).device)
        inputs = Variable(features,requires_grad=False).view(-1, 1, features.shape[1]).to(device)
        #print("@@@@")
        #print(inputs.shape)

        probabilities = output_transform(model(inputs))
        #print("try")
        #print(*probabilities.shape[1:])

        probabilities = (
            probabilities
            .view(-1, K, *probabilities.shape[1:])
            .mean(dim=1)
        )
        #print(probabilities.shape)
        probabilities = sharpen(probabilities, T) #.to(original_device)
        #print("@#@#@#")
        #print(probabilities.shape)

    return tile_adjacent(probabilities, K)
    #return probabilities