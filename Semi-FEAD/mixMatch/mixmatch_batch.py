import torch
from guess_targets import guess_targets
from mixup_samples import mixup_samples
import numpy as np


def mixmatch_batch(
        batch,batch_unlabeled, model, output_transform, K, T, beta
):
    features_labeled = batch['features']
    targets_labeled = batch['targets']
    features_unlabeled = batch_unlabeled[0] # ['features']
    #print("!!!!")
    from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse
    my_augmenter = (
            TimeWarp()
            + Crop(size=155)
            + Quantize(n_levels=[10, 20, 30])
            + Drift(max_drift=(0.1, 0.5))
            #+ Reverse()
    )
    # new = np.zeros((16,155))
    new = None
    num = 0
    for u in features_unlabeled:
        #print(len(u.numpy().tolist()))
        x = my_augmenter.augment(u.numpy())
        if new is None:
            new = x.unsqueeze(0)
        else:
            new = np.stack(new, x.unsqueeze(0))
        num = num + 1
    features_unlabeled = torch.from_numpy(new.astype(np.float32))

    #print(features_unlabeled.shape)
    #print(features_labeled.shape)



    targets_unlabeled = guess_targets(
        features_unlabeled, model, output_transform, K, T
    )

    #print("!!")
    #print(features_unlabeled.shape)
    #print(targets_unlabeled.shape)

    indices = torch.randperm(len(features_labeled) + len(features_unlabeled))
    features_W = torch.cat((features_labeled, features_unlabeled), dim=0)[indices]
    #print(targets_labeled.shape,"!!",targets_unlabeled.shape)

    targets_W = torch.cat((targets_labeled, targets_unlabeled), dim=0)[indices]

    features_X, targets_X = mixup_samples(
        features_labeled,
        targets_labeled,
        features_W[:len(features_labeled)],
        targets_W[:len(features_labeled)],
        beta
    )
    features_U, targets_U = mixup_samples(
        features_unlabeled,
        targets_unlabeled,
        features_W[len(features_labeled):],
        targets_W[len(features_labeled):],
        beta
    )

    return dict(
        features=torch.cat((features_X, features_U), dim=0),
        targets=torch.cat((targets_X, targets_U), dim=0),
    )
