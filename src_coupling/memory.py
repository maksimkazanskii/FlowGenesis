import torch

class SpectralMemory:

    def __init__(self,n_classes,dim,device):

        self.centroids=torch.zeros(
            n_classes,
            dim,
            device=device
        )

        self.counts=torch.zeros(
            n_classes,
            device=device
        )

        self.frozen=set()

    def freeze(self,classes):

        for c in classes:
            self.frozen.add(int(c))

    @torch.no_grad()
    def update(self,spectrum,labels):

        for c in labels.unique():

            c=int(c)

            if c in self.frozen:
                continue

            m=labels==c

            mean=spectrum[m].mean(0)

            old=self.counts[c]
            new=m.sum()

            self.centroids[c]=(
                                      old*self.centroids[c]
                                      + new*mean
                              )/(old+new)

            self.counts[c]+=new