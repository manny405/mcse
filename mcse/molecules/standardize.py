
from mcse import com
from mcse.molecules.align import fast_align
from mcse.molecules.label import label


def standardize(struct):
    fast_align(struct)
    label(struct)
    return struct