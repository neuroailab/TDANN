"""
Loads data from Ringach et al., 2002:
    Orientation selectivity in macaque V1: diversity and laminar dependence.

The website (ringachlab.net) isn't super reliable, so here's the documentation there:

>  This database describes the tuning curves of 308 neurons in macaque V1
>  (M. fascicularis). This are the raw data for all the figures in the paper
>  Ringach et al, “Orientation Selectivity in Macaque V1: Diversity and
>  Laminar Dependence”. J Neurosci. 22(13):5639-5651, 2002.  The variables in the
>  file are as follows:


>  db.orivar: circular variance of the orientation tuning curve based on mean spike rate
>  db.maxdc: maximum spike rate across all orientations
>  db.mindc: minimum spike rate across all orientations
>  db.maxfirst: the first harmonic response at the optimal orientation
>  db.po: The ratio between the responses at the orthogonal vs preferred orientation
>  db.bw: Bandwidth of the tuning curve
>  db.spont: Spontaneous firing rate
>  db.depth: Normalized cortical distance (see paper for explanation)
>  db.animal: Animal ID
>  db.expt: Experiment ID
>  db.chan: Electrode channel
>
>  Funding: Work funded by NIH-EY12816, NSF-IBN-9720305, EY-08300, EY-01472
"""

from pathlib import Path
import numpy as np
from scipy.io import loadmat


def load_ringach_data(fieldname=None, squeeze=True):
    """
    Loads Ringach data from matfile and formats it to a
    dictionary with named fields

    Inputs
        fieldname (str): the name of the field to return. If None, returns
            the entire dict
        squeeze (bool): if True, calls np.squeeze before returning output.
            Ignored if no fieldname is provided
    """
    module_dir = Path(__file__).parent
    file_path = str(module_dir / "ringach_2002_db.mat")
    loaded_mat = loadmat(file_path)
    data = loaded_mat["db"]
    dtypes = data.dtype
    data_dict = {name: data[name][0, 0] for name in dtypes.names}
    if fieldname is not None:
        if squeeze:
            return np.squeeze(data_dict[fieldname])
        return data_dict[fieldname]
    return data_dict
