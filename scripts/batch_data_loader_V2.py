"""
This class has the function to ensure that in every batch, where is at least one fake image
"""
import random
from torch.utils.data import Sampler
from typing import List, Iterator
import numpy as np

class BatchPatternSampler(Sampler):
    def __init__(self, fake_indices, real_indices, num_batch, batch_size, epoch):
        self.fake_indices = list(fake_indices)
        self.real_indices = list(real_indices)
        if batch_size != 2:
            raise ValueError("batch_size must be 2 ")
        if len(fake_indices) == 0:
            raise ValueError("Need at least 1 fake index to guarantee 'at least one fake per batch'.")
        if len(real_indices) == 0:
            raise ValueError("Need at least 1 real index to guarantee 'at least one fake per batch'.")
        if ((len(fake_indices) + len(real_indices)) ) != 2 * num_batch:
            raise ValueError("num fake + num real != batch_size * 2")
        if (len(fake_indices) < num_batch):
            raise ValueError("num fake needs to be higher than the number of batches")
        
        self.epoch = epoch
        self.num_batch = num_batch 
        self.rest_fake = len(fake_indices) - num_batch

        self.pattern = [2] * self.rest_fake + [1] * len(real_indices) # fake : 2 real:1
        
        
        self.i_fake = 0
        self.i_real = 0
        
    def __len__(self) -> int:
        # BatchSampler-Länge = Anzahl Batches pro Epoche
        return self.num_batch
    
    def __iter__(self) -> Iterator[List[int]]:
        # per epoch deterministic
        rng = random.Random(self.epoch)

        # local copy
        fake = self.fake_indices.copy()
        real = self.real_indices.copy()
        rng.shuffle(fake)
        rng.shuffle(real)
        rng.shuffle(self.pattern)

        self.i_fake = 0
        self.i_real = 0

        for b in range(self.num_batch):
            batch: List[int] = []
            rng_batch = random.Random(self.epoch + b)
            order_fake_first = rng_batch.random() < 0.5

            if order_fake_first:
                batch.append(self._take_fake(fake))
                if self.pattern[b] == 1:
                    batch.append(self._take_real(real))
                else:
                    batch.append(self._take_fake(fake))
            else:
                if self.pattern[b] == 1:
                    batch.append(self._take_real(real))
                else:
                    batch.append(self._take_fake(fake))

                batch.append(self._take_fake(fake))

            yield batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

        


    def _take_fake(self, fake):
        # zyklisch/falls leer: reshuffle + reset
        if self.i_fake >= len(fake):
            raise ValueError(f"length of trian fake data {len(fake)} exeded with i_fake: {self.i_fake}")
        
        idx = fake[self.i_fake]
        self.i_fake += 1
        return idx  

    def _take_real(self, real):
        # zyklisch/falls leer: reshuffle + reset
        if self.i_real >= len(real):
            raise ValueError(f"length of train real data {len(real)} exeded with i_real: {self.i_real}")
        
        idx = real[self.i_real]
        self.i_real += 1
        return idx           