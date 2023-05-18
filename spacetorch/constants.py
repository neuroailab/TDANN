import numpy as np

# how many degrees is the visual input?
DVA_PER_IMAGE = 8.0

# angles and spatial frequencies used most often
DEFAULT_ANGLES = np.linspace(0, 180, 9)[:-1]  # degrees
DEFAULT_SFS = np.linspace(5.5, 110.0, 9)[:-1]  # cycles per image
DEFAULT_PHASES = np.linspace(0, 2 * np.pi, 6)[:-1]

# estimate of the size of the retina sensitive to central eight degrees, given an
# estimate of roughly 300 microns per visual degree
RETINA_SIZE = 2.4  # mm

# estimate of how big, in mm, V1 in a single hemisphere is, from Benson et al
#  (sqrt 13.5cm^2 = 36.75mm)
V1_SIZE = 36.75  # mm

# V2 estimate also from Benson et al, slightly smaller than V1
V2_SIZE = 35.0  # mm

# human V4 estimates are hard to find, but this seems like a reasonable approximation
V4_SIZE = 22.4  # mm

# this is measured from the average size of the responsive VTC ROI in the NSD
VTC_SIZE = 70.0  # mm

# for good luck!
RNG_SEED = 424
