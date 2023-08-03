# privECG

This is the repository for the paper: PrivECG: generating private ECG for end-to-end anonymization. This paper was presented during the 2023 edition of MLHC. 

## Setup

```
pip install requirements.txt
```

## Important note

We decided to used 12-lead ECG in the format (500,12) by downsampling the original data. As usually, depending on sampling frequency, ECG data varies from length 5000 to 1250, we propose to simply downsample to a length 500. This could be done with a function such as this one:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.interpolate import interp1d
from tqdm import tqdm

def ResampleLinear1D(original, targetLen):
    original = np.array(original, dtype=np.float)
    index_arr = np.linspace(0, len(original)-1, num=targetLen, dtype=np.float)
    index_floor = np.array(index_arr, dtype=np.int) #Round down
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor #Remain

    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    interp = val1 * (1.0-index_rem) + val2 * index_rem
    assert(len(interp) == targetLen)
    return interp

for ECG_pos,ECG in enumerate(tqdm(dataset)):
  for lead_pos,lead in enumerate(ECG):
    dataset[ECG_pos][lead_pos] = ResampleLinear1D(lead,500)
```

An even more naive version could simply be

```python
#example for original len of 2500
for ECG_pos,ECG in enumerate(tqdm(dataset)):
  for lead_pos,lead in enumerate(ECG):
    dataset[ECG_pos][lead_pos] = lead[1::5]
```

It must be also noted that we require all the ECG leads need to be stadardized between 0 and 1. We do note that this could affect the downstream utility as relative peak height variation accross leads can be informative of various diseases. However, for better GAN performance we suggest standardizing in a per-lead manner.

## The Data

Sadly, the data used for this study is not publicly available as it was not part of the original release form while also going against the heart of the paper.

## Next steps

I think it would be cool to not attempt to naively induce controled noise but first try to induce privacy by simply varying rhythm thus requiring less transformation allowing better readability while rendering re-identification sufficiently hard. More on this soon hopefully ...

