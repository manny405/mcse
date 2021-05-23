# -*- coding: utf-8 -*-

import numpy as np

from matplotlib.ticker import FormatStrFormatter

from matplotlib.pyplot import cm
import matplotlib.colors as clr


def correct_energy(energy_list, nmpc=4, global_min=None):
    '''
    Purpose:
        Corrects all energies values in the energy list relative to the global
          min and converts eV to kJ/mol/molecule.
    '''
    if global_min == None:
        global_min = min(energy_list)
    corrected_list = []
    for energy_value in energy_list:
        corrected_energy = energy_value - global_min
        corrected_energy = corrected_energy/(0.010364*float(nmpc))
        corrected_list.append(corrected_energy)
    return corrected_list



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=10000):
    """
    Truncates the colormap between the min value and the max value with n 
    divisions. This is useful for removing values near the edges of some 
    colormaps that are too light or too dark for easy viewing.
    
    Argumnets
    ---------
    cmap: matplotlib.colormap
        Matplotlib colormap to use to truncate
    minval: float
        Float between 0 < minval < maxval
    maxval: float
        Float such that minval < maxval < 1.0
    n: int
        Number of divisions to create in colormap.
    
    """
    new_cmap = clr.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def colorbar(mappable):
    """
    Adds colormap to the mappable axis such that the colormap is the perfect 
    size for the graph.
    
    """
    ## From: https://joseph-long.com/writing/colorbars/
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax,
                        ticklocation="top",
                        )
    plt.sca(last_axes)
    return cbar
    

def colors_from_colormap(num, cmap):
    """
    Returns colors that are evenly spaced along a matplotlib colormap for plotting 
    two or more colored objects on a single plot. This is useful if the user is 
    expecting to plot many lines together and wants them all to be distinct in
    color. Different colormaps can be used to give more or less distinction
    between the lines giving the user a lot of control over the colors very 
    easily. 
    
    Argumnets
    ---------
    num: int
        Number of colors to return. 
    cmap: matplotlib.colormap
        The colormap to use to obtain colors. 
    
    """
    ## First check if colormap is discrete. Best way I can see to do it is 
    ## through the colors attribute
    if getattr(cmap, "colors", False):
        ### Get maximally different colors
        num_colors_cmap = len(cmap.colors)
        skip_number = int(num_colors_cmap / num)
        return cmap.colors[::skip_number][:num]
        
    ints = np.arange(0,num+1,1)
    norm = clr.Normalize(vmin=ints[0], vmax=ints[-1], clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    color_list = []
    for i in ints:
        color_list.append(mapper.to_rgba(i))
    
    return color_list


def colors_from_colormap_continuous(num, cmap):
    """
    Returns colors that are evenly spaced along a matplotlib colormap for plotting 
    two or more colored objects on a single plot. This is useful if the user is 
    expecting to plot many lines together and wants them all to be distinct in
    color. Different colormaps can be used to give more or less distinction
    between the lines giving the user a lot of control over the colors very 
    easily. 
    
    Argumnets
    ---------
    num: int
        Number of colors to return. 
    cmap: matplotlib.colormap
        The colormap to use to obtain colors. 
    
    """ 
    ints = np.arange(0,num+1,1)
    norm = clr.Normalize(vmin=ints[0], vmax=ints[-1], clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    color_list = []
    for i in ints:
        color_list.append(mapper.to_rgba(i))
    
    return color_list




def labels_and_ticks(
    ax,
    xlabel_kw = 
        {
            "xlabel": "Unit Cell Volume, $\AA^3$",
            "fontsize": 16,
            "labelpad": 0,
        },
    ylabel_kw = 
        {
            "ylabel": "Number of Structures",
            "fontsize": 16,
            "labelpad": 10, 
        },
    xticks = 
        {
            "xlim": [],
            "xticks_kw":
                {
                    "ticks": [],
                },
            "xticklabels_kw": 
                {
                    "labels": [],
                    "fontsize": 12,
                },
            "FormatStrFormatter": "%.2f"
        },
    yticks = 
        {
            "ylim": [],
            "yticks_kw":
                {
                    "ticks": [],
                },
            "yticklabels_kw": 
                {
                    "labels": [],
                    "fontsize": 12,
                },
            "FormatStrFormatter": "%.2f"
        },
    ):
    """
    Primitive function for handling the assignment of axis labels tick values.
    It's common for all graphs to require this step, so may as well use a
    signle function for it. 
    
    """    
    ax.set_xlabel(**xlabel_kw)
    ax.set_ylabel(**ylabel_kw)
    
    #### Dealing with xticks and x labels
    if len(xticks["xticks_kw"]["ticks"]) > 0:
        ax.set_xticks(**xticks["xticks_kw"])
    # Otherwise, add the current xtick values to the argument dictionary
    else:
        try: xticks["xticks_kw"]["ticks"] = ax.get_xticks().tolist()
        except: xticks["xticks_kw"]["ticks"] = ax.get_xticks()
        
    if len(xticks["xticklabels_kw"]["labels"]) > 0:
        ax.set_xticklabels(**xticks["xticklabels_kw"])
    else:
        # Set with current xtick values. Otherwise, the xticklabels are empty
        # until the graph is draw.
        del(xticks["xticklabels_kw"]["labels"])
        ax.set_xticklabels(labels=ax.get_xticks(),
                           **xticks["xticklabels_kw"])
        xticks["xticklabels_kw"]["labels"] = [x.get_text() for x in 
                                              ax.get_xticklabels()]
    
    if len(yticks["yticks_kw"]["ticks"]) > 0:
        ax.set_yticks(**yticks["yticks_kw"])
    else:
        yticks["yticks_kw"]["ticks"] = ax.get_yticks().tolist()
        
    if len(yticks["yticklabels_kw"]["labels"]) > 0:
        ax.set_yticklabels(**yticks["yticklabels_kw"])
    else:
        # Set with current ytick values. Otherwise, the yticklabels are empty
        # until the graph is draw.
        del(yticks["yticklabels_kw"]["labels"])
        ax.set_yticklabels(labels=ax.get_yticks(),
                           **yticks["yticklabels_kw"])
        yticks["yticklabels_kw"]["labels"] = [x.get_text() for x in 
                                              ax.get_yticklabels()]

    # Deal with the limits of the graph    
    if len(yticks["ylim"]) == 0:
        ax.autoscale()
        yticks["ylim"] = ax.get_ylim()
    if len(xticks["xlim"]) == 0:
        ax.autoscale()
        xticks["xlim"] = ax.get_xlim()
    ax.set_ylim(yticks["ylim"])
    ax.set_xlim(xticks["xlim"])
    
    if xticks.get("FormatStrFormatter"):
        ax.xaxis.set_major_formatter(
                FormatStrFormatter(xticks["FormatStrFormatter"]))
    if yticks.get("FormatStrFormatter"):
        ax.yaxis.set_major_formatter(
                FormatStrFormatter(yticks["FormatStrFormatter"]))
        
def label_and_ticks_right_yaxis(
        ax,
        ylabel_kw_right = 
            {
                "ylabel": "",
                "fontsize": 16,
                "labelpad": 10, 
            },
        yticks_right = 
            {
                "ylim": [],
                "yticks_kw":
                    {
                        "ticks": [],
                    },
                "yticklabels_kw": 
                    {
                        "labels": [],
                        "fontsize": 12,
                    },
                "FormatStrFormatter": "%.2f"
            }):
    """
    For formatting only the y-axis values for the y-axis on the right-hand side
    of the plot. 
    
    """
    ax.set_ylabel(**ylabel_kw_right)
    
    if len(yticks_right["yticks_kw"]["ticks"]) > 0:
        ax.set_yticks(**yticks_right["yticks_kw"])
    else:
        yticks_right["yticks_kw"]["ticks"] = ax.get_yticks().tolist()
        
    if len(yticks_right["yticklabels_kw"]["labels"]) > 0:
        ax.set_yticklabels(**yticks_right["yticklabels_kw"])
    else:
        # Set with current ytick values. Otherwise, the yticklabels are empty
        # until the graph is draw.
        del(yticks_right["yticklabels_kw"]["labels"])
        ax.set_yticklabels(labels=ax.get_yticks(),
                           **yticks_right["yticklabels_kw"])
        yticks_right["yticklabels_kw"]["labels"] = [x.get_text() for x in 
                                                    ax.get_yticklabels()]

    # Deal with the limits of the graph    
    if len(yticks_right["ylim"]) == 0:
        ax.autoscale()
        yticks_right["ylim"] = ax.get_ylim()
    
    if yticks_right.get("FormatStrFormatter"):
        ax.yaxis.set_major_formatter(
                FormatStrFormatter(yticks_right["FormatStrFormatter"]))
        
        
def labels_and_ticks_3D(
    ax,
    xlabel_kw = 
        {
            "xlabel": "Unit Cell Volume, $\AA^3$",
            "fontsize": 16,
            "labelpad": 0,
        },
    ylabel_kw = 
        {
            "ylabel": "Number of Structures",
            "fontsize": 16,
            "labelpad": 10, 
        },
    zlabel_kw = 
        {
            "ylabel": "c, $\AA$",
            "fontsize": 16,
            "labelpad": 10, 
        },
    xticks = 
        {
            "xlim": [],
            "xticks_kw":
                {
                    "ticks": [],
                },
            "xticklabels_kw": 
                {
                    "labels": [],
                    "fontsize": 12,
                },
            "FormatStrFormatter": "%.2f"
        },
    yticks = 
        {
            "ylim": [],
            "yticks_kw":
                {
                    "ticks": [],
                },
            "yticklabels_kw": 
                {
                    "labels": [],
                    "fontsize": 12,
                },
            "FormatStrFormatter": "%.2f"
        },
    zticks = 
        {
            "zlim": [],
            "zticks_kw":
                {
                    "ticks": [],
                },
            "zticklabels_kw": 
                {
                    ## Unfortunately, this is named differently and not 
                    ## consistent with x and y
                    "ticklabels": [],
                    "fontsize": 12,
                },
            "FormatStrFormatter": "%.2f"
        } 
    ):
    """
    Primitive function for handling the assignment of axis labels tick values.
    It's common for all graphs to require this step, so may as well use a
    signle function for it. 
    
    """    
    ax.set_xlabel(**xlabel_kw)
    ax.set_ylabel(**ylabel_kw)
    ax.set_zlabel(**zlabel_kw)
    
    # Deal with the limits of the graph    
    if len(yticks["ylim"]) == 0:
        ax.autoscale()
        yticks["ylim"] = ax.get_ylim()
    if len(xticks["xlim"]) == 0:
        ax.autoscale()
        xticks["xlim"] = ax.get_xlim()
    if len(zticks["zlim"]) == 0:
        ax.autoscale()
        zticks["zlim"] = ax.get_zlim()
    ax.set_ylim(yticks["ylim"])
    ax.set_xlim(xticks["xlim"])
    ax.set_zlim(zticks["zlim"])
    
    #### Dealing with xticks and x labels
    if len(xticks["xticks_kw"]["ticks"]) > 0:
        ax.set_xticks(**xticks["xticks_kw"])
    # Otherwise, add the current xtick values to the argument dictionary
    else:
        try: xticks["xticks_kw"]["ticks"] = ax.get_xticks().tolist()
        except: xticks["xticks_kw"]["ticks"] = ax.get_xticks()
        
    if len(xticks["xticklabels_kw"]["labels"]) > 0:
        ax.set_xticklabels(**xticks["xticklabels_kw"])
    else:
        # Set with current xtick values. Otherwise, the xticklabels are empty
        # until the graph is draw.
        ax.set_xticklabels(ax.get_xticks())
        xticks["xticklabels_kw"]["labels"] = [x.get_text() for x in 
                                              ax.get_xticklabels()]
    
    if len(yticks["yticks_kw"]["ticks"]) > 0:
        ax.set_yticks(**yticks["yticks_kw"])
    else:
        yticks["yticks_kw"]["ticks"] = ax.get_yticks().tolist()
        
    if len(yticks["yticklabels_kw"]["labels"]) > 0:
        ax.set_yticklabels(**yticks["yticklabels_kw"])
    else:
        # Set with current ytick values. Otherwise, the yticklabels are empty
        # until the graph is draw.
        ax.set_yticklabels(ax.get_yticks())
        yticks["yticklabels_kw"]["labels"] = [x.get_text() for x in 
                                              ax.get_yticklabels()]
    
    if len(zticks["zticks_kw"]["ticks"]) > 0:
        ax.set_zticks(**zticks["zticks_kw"])
    else:
        zticks["zticks_kw"]["ticks"] = ax.get_zticks().tolist()
    
    if len(zticks["zticklabels_kw"]["labels"]) > 0:
        ax.set_zticklabels(**zticks["zticklabels_kw"])
    else:
        ax.set_zticklabels(ax.get_zticks())
        zticks["zticklabels_kw"]["ticklabels"] = [x.get_text() for x in 
                                              ax.get_zticklabels()]
    
    if xticks.get("FormatStrFormatter"):
        ax.xaxis.set_major_formatter(
                FormatStrFormatter(xticks["FormatStrFormatter"]))
    if yticks.get("FormatStrFormatter"):
        ax.yaxis.set_major_formatter(
                FormatStrFormatter(yticks["FormatStrFormatter"]))
    if zticks.get("FormatStrFormatter"):
        ax.zaxis.set_major_formatter(
                FormatStrFormatter(zticks["FormatStrFormatter"]))


# Default formating parameters
d_label_size=14
d_tick_size=12
d_line_width = 3
d_figure_size=(10,8)
d_labelpad_x = 5
d_labelpad_y1 = 5
d_labelpad_y2 = 20
d_tick_width=2
d_capsize=0
d_errorevery=10
d_elinewidth=0
d_grid=False

###############################################################################
# Aesthetics                                                                  # 
###############################################################################

def format_ticks(ax,
                   linewidth=d_line_width,
                   tick_width=d_tick_width,
                   tick_size=d_tick_size,
                   grid=d_grid):
    ax.spines['top'].set_linewidth(tick_width)
    ax.spines['right'].set_linewidth(tick_width)
    ax.spines['bottom'].set_linewidth(tick_width)
    ax.spines['left'].set_linewidth(tick_width)
    ax.tick_params(axis='both', which='major', labelsize=tick_size,
                   width=2, length=7)
    ax.grid(grid, axis='both', which='both')
