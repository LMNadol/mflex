import numpy as np
import matplotlib.pyplot as plt

from bfieldmodel import b3d, b3d_2f1, b3d_io


class Field3d:

    def __init__(self, path2file: str, a, b, alpha, figure=True, fieldlines='all', footpoints='grid', addmodel=None):

        self.path2file = path2file 
        self.photo = np.load(path2file)

        self.field3d = b3d(path2file, a, b, alpha)

        self.figure = plt.figure()

        if figure is True:
            self.add_magnetogram(path2file)
            self.add_fieldlines()

    def add_magnetogram(self, path2file): 
        # add photospheric magnetogram to plot
    

    def add_fieldlines(self, *args, **kwargs):
        # add field lines to plot
