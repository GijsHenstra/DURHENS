# -----------------------------------------------------------------------------
# Copyright (c) 2020 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
# This example shows how to render bars
# -----------------------------------------------------------------------------
from mpl3d import glm
from mpl3d.camera import Camera
import numpy as np

import math

# from copy import copy

import cv2

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# import help_functions as hf

# import line_profiler
# profile = line_profiler.LineProfiler()

class Bar:
    """ Bar (histogram). """
    # @profile
    def __init__(self, ax, transform, Z,
                 facecolors="white", edgecolors="black",
                 linewidth=0, shadefaces=None, clip=False):
        """ """


        self.Z = Z
        
        if isinstance(facecolors, np.ndarray):
            shape = facecolors.shape
            facecolors = facecolors.reshape(-1, shape[-1])
            facecolors = mpl.colors.to_rgba_array(facecolors)
            self.facecolors = facecolors.reshape(shape[0], shape[1], 4)
        else:
            shape = Z.shape
            self.facecolors = np.zeros((shape[0], shape[1], 4))
            self.facecolors[...] = mpl.colors.to_rgba(facecolors)

        if isinstance(edgecolors, np.ndarray):
            shape = edgecolors.shape
            edgecolors = edgecolors.reshape(-1, shape[-1])
            edgecolors = mpl.colors.to_rgba_array(edgecolors)
            self.edgecolors = edgecolors.reshape(shape[0], shape[1], 4)
        else:
            shape = Z.shape
            self.edgecolors = np.zeros((shape[0], shape[1], 4))
            self.edgecolors[...] = ((mpl.colors.to_rgba(edgecolors)))

        self.linewidth = linewidth
        self.xlim = -0.5, +0.50
        self.ylim = -0.5, +0.50
        self.zlim = -0.5, +0.50

        self.clip = clip

        # Because all the bars have the same orientation, we can use a trick to
        # shade each face at once instead of computing individual face lighting.


        if isinstance(shadefaces, np.ndarray):
            if len(shadefaces) == 6:
                self.shadefaces = shadefaces[np.newaxis, :]
            if len(shadefaces) == 1:
                self.shadefaces = shadefaces
        else:
            # standard shadowing
            self.shadefaces = np.array([[0.95, 0.6, 0.75, 0.91, 0.850, 0.95]])


        self.collection = PolyCollection([], clip_on=self.clip, snap=False)
        self.update(transform)
        ax.add_collection(self.collection, autolim=False)


    # @profile
    def update(self, transform):
        """ """

        Z = self.Z
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        zmin, zmax = self.zlim
        dx, dy = 1.07 * 1 / Z.shape[0], 1 * 1.07 / Z.shape[1]

        # Each bar is described by 8 vertices and 6 faces
        V = np.zeros((Z.shape[0], Z.shape[1], 8, 3))
        F = np.zeros((Z.shape[0], Z.shape[1], 6, 4), dtype=int)

        # Face and edge colors for the six faces
        FC = np.zeros((Z.shape[0], Z.shape[1], 6, 4))
        FC[:, :] = self.facecolors.reshape(Z.shape[0], Z.shape[1], 1, 4)
        FC *= self.shadefaces.T

        FC[:, :, :, 3] = 1
        
        nan_rgb = 0.9
        FC[np.isnan(Z)] = nan_rgb
        
        nan_alpha = 1
        FC[np.isnan(Z), ..., 3] = nan_alpha


        EC = np.zeros((Z.shape[0], Z.shape[1], 6, 4))
        EC[:, :] = self.edgecolors.reshape(Z.shape[0], Z.shape[1], 1, 4)

        # Build vertices
        X, Y = np.meshgrid(np.linspace(xmin, xmax, Z.shape[0]),
                           np.linspace(ymin, ymax, Z.shape[1]))
        V[..., 0] = X.reshape(Z.shape[0], Z.shape[1], 1)
        V[..., 1] = Y.reshape(Z.shape[0], Z.shape[1], 1)

        V[:, :, 0] += [+dx/2, +dy/2, zmin]
        V[:, :, 1] += [+dx/2, -dy/2, zmin]
        V[:, :, 2] += [-dx/2, -dy/2, zmin]
        V[:, :, 3] += [-dx/2, +dy/2, zmin]

        V[:, :, 4] += [+dx/2, +dy/2, zmin]
        V[:, :, 5] += [+dx/2, -dy/2, zmin]
        V[:, :, 6] += [-dx/2, -dy/2, zmin]
        V[:, :, 7] += [-dx/2, +dy/2, zmin]

        V[:, :, 4:, 2] += Z.reshape(Z.shape[0], Z.shape[1], 1)

        # Build faces
        I = 8 * np.arange(Z.shape[0] * Z.shape[1])
        F[:, :] = I.reshape(Z.shape[0], Z.shape[1], 1, 1)
        F[:, :] += [[0, 1, 2, 3],  # -Z
                    [0, 1, 5, 4],  # +X
                    [2, 3, 7, 6],  # -X
                    [1, 2, 6, 5],  # -Y
                    [0, 3, 7, 4],  # +Y
                    [4, 5, 6, 7]]  # +Z

        # Actual transformation
        V = V.reshape(-1, 3)
        V = glm.transform(V[F], transform)  # [...,:2]

        # Depth computation
        # We combine the global "depth" of the bar (depth of the bottom face)
        # and the local depth of each face. This trick avoids problems when
        # sorting all the different faces.
        Z1 = (V[:, :, 0, :, 2].mean(axis=2)).reshape(Z.shape[0], Z.shape[1], 1)
        Z2 = (V[..., 2].mean(axis=3) + 10 * Z1).ravel()

        # Sorting
        I = np.argsort(-Z2)
        V = (V[..., :2].reshape(Z.shape[0] * Z.shape[1] * 6, 4, 2))

        # print("V[I]", V[I])
        # print('EC.reshape(-1, 4)[I]',EC.reshape(-1, 4)[I])

        self.collection.set_verts(V[I])
        self.collection.set_facecolors(FC.reshape(-1, 4)[I])
        self.collection.set_edgecolors(EC.reshape(-1, 4)[I])
        self.collection.set_linewidths(self.linewidth)
        if self.linewidth == 0.0:
            self.collection.set_antialiased(True) # before, was False
        else:
            self.collection.set_antialiased(True)


def illumination_faces(
        azi_rad, zen_rad, beer_lambert=False, min_visibility=0):
    """Calculate the ratio of sun exposure (0 to 1) on faces of an element."""
    # intensity = np.array([math.cos(zen_rad),
    #                       math.sin(zen_rad) * math.sin(azi_rad),
    #                       math.sin(zen_rad) * math.sin(azi_rad + math.pi),
    #                       math.sin(zen_rad),
    #                       math.sin(zen_rad) * math.cos(azi_rad + math.pi),
    #                       math.cos(zen_rad)])

    intensity = np.array([math.cos(zen_rad),  # "Bottom", or sometimes also the top
                          math.sin(zen_rad) * math.sin(azi_rad),  # "East"
                          math.sin(zen_rad) * \
                          math.sin(azi_rad + math.pi),  # "West"
                          math.sin(zen_rad) * math.cos(azi_rad),  # " North"
                          math.sin(zen_rad) * \
                          math.cos(azi_rad + math.pi),  # "South"
                          math.cos(zen_rad)])  # "Top"
        
    if math.degrees(zen_rad) > 90:
        intensity *= 0

    if beer_lambert:
        try:
            intensity *= math.exp(-0.025 / math.cos(zen_rad))
        except OverflowError:
            intensity *= 0

    intensity = np.where(zen_rad > math.pi / 2, 0, intensity)

    intensity = np.clip(intensity, a_min=0, a_max=1)

    if min_visibility > 0:
        intensity = min_visibility * \
            np.array([1, 0.92, 0.95, 0.9, 0.97, 1]) + \
            (1 - min_visibility) * intensity

    return intensity

# @profile
def plot(height2d,
         gridsize,
         view_zen,
         view_azi,
         img=None,
         bars=None,
         contours=None,
         contours_rgb=(0, 0, 0),
         colorbar_lims=None,
         res=1,
         # shade=None,
         shadefaces=None,
         linewidth=0,
         figtitle='',
         colorbar=None,
         aspect='equal',
         std_color='whitesmoke',
         cmap='viridis',
         save_path='barplot',
         save_ext='.png',
         saveimg=False,
         dpi=300,
         plot=True,
         cam_scale=1,
         mode="perspective",
         colorbar_extend='neither',
         ax=None,
         **kwargs):
    
    Z = np.abs(height2d)
    
    if 'clim' in kwargs:
        clim = kwargs['clim']
    
    else:
        if img.shape == height2d.shape:
            clim = (np.nanmin(img), np.nanmax(img))
        else:
            if (height2d < 0).any():
                max_val = max(abs(np.nanmin(height2d)), abs(np.nanmax(height2d)))
                clim = (-max_val, max_val)
            else:
                clim = (np.nanmin(height2d), np.nanmax(height2d))
            
    norm = mpl.colors.Normalize(vmin=clim[0], 
                                vmax=clim[1])
    
    
        
    cmap = plt.get_cmap(cmap)
    if img is None:
        
        img = cmap(norm(height2d))
    else:
        if img.shape == height2d.shape:
            img = cmap(img)            
            # img = cmap(norm(img))

        
    if (img > 1).any(): 
        img = img/255
    
    if contours is not None:
        img = cv2.drawContours(img.copy(), contours, -1, contours_rgb, 1)
            
    
    # import warnings
    azi_corr = 1 + (0.35 * abs(np.sin(np.deg2rad(view_azi * 2))))
    zen_corr = 0.2 * np.cos(np.deg2rad(view_zen))

    xlim = list(azi_corr * (0.7 - zen_corr)
                * (np.array([-1, 1]) + (-0.05 if colorbar else +0)))
    ylim = list(azi_corr * (0.7 - zen_corr) *
                (np.array([1, -1]) + 0.5 * np.sin(np.deg2rad(view_zen))**(1 / 2)))

    if ax is None:
        fig = plt.figure(figsize=(20, 6))
        ax = fig.add_axes([0, 0, 1, 1], xlim=xlim, ylim=ylim, aspect=1)
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect(1)

    ax.axis("off")
    if isinstance(colorbar, mpl.cm.ScalarMappable):
        plt.colorbar(colorbar, ax=ax, extend=colorbar_extend, fraction=0.0415, pad=0.04)
        
    
    
    
    if aspect == 'equal':
        Z_norm = Z / gridsize
    else:
        if colorbar == True:
            if 'sm' in kwargs:
                sm = kwargs['sm']
            else:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
            plt.colorbar(mappable=sm, ax=ax, extend=colorbar_extend, fraction=0.0415, pad=0.04)
        if isinstance(aspect, (float, int)):
            Z_norm = Z / np.nanmax(Z) * aspect
    camera = Camera(mode, -view_zen + 0.01, view_azi + 0.01, scale=cam_scale)
    bars = Bar(ax, camera.transform, Z_norm,
               shadefaces=shadefaces, facecolors=img, linewidth=linewidth)

    camera.connect(ax, bars.update)

    if save_path is not None and saveimg:
        plt.savefig(save_path + save_ext, dpi=dpi, bbox_inches='tight')
    
    if len(figtitle) > 0:
        ax.set_title(figtitle)
        
    if plot:
        plt.show(block=False)
    else:
        plt.close()

    # plt.close()
    
    return bars


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # import imageio

    # from matplotlib.patches import Circle

    # Z = imageio.imread("rotate_me.png")[::10,::10,0]
    Z = chm_round_lr[:30, :30]
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    Z += 0.01 * np.random.uniform(0, 1, Z.shape)
    Z = 0.25 * Z
    # Z = 0.25*Z*Z

    cmap = plt.get_cmap("Reds")
    norm = mpl.colors.Normalize(vmin=Z.min(), vmax=Z.max())
    facecolors = cmap(norm(Z))

    fig = plt.figure(figsize=(10, 5))
    # fig, axs = plt.subplots(2, 2, figsize=(6, 9))
    ax = fig.add_axes([0, 0, 1, 1], xlim=[-1, +1], ylim=[-1, 0], aspect=1)
    ax.axis("off")

    camera = Camera("perspective", 65, -125)
    bars = Bar(ax, camera.transform, Z, facecolors=facecolors)
    camera.connect(ax, bars.update)

    plt.savefig("bar.png", dpi=300)
    plt.show()
