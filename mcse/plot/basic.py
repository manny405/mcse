# -*- coding: utf-8 -*-


import copy
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors 

from mcse.plot.utils import labels_and_ticks,\
                              label_and_ticks_right_yaxis, \
                              truncate_colormap,\
                              colors_from_colormap


def text_plot(
        text="",
        text_kw =
            {
                "x": 0.025,
                "y": 0.5,
                "horizontalalignment": "left",
                "fontsize": 12,
                "wrap": True,
            },
        ax=None,
        figname="",
        fig_kw={}):
    arguments = locals()
    arguments_copy = {}
    for key,value in arguments.items():
        if key == "ax":
            arguments_copy[key] = value
        else:
            arguments_copy[key] = copy.deepcopy(value)
    arguments = arguments_copy
    
    fig = None
    if ax == None:
        fig = plt.figure(**arguments["fig_kw"])
        ax = fig.add_subplot(111)
    
    text_kw["s"] = text
    ax.text(**text_kw)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    ## Save if figname provided
    if len(figname) > 0:
        if fig == None:
            fig = plt.figure(**arguments["fig_kw"])
            temp_ax = fig.add_subplot(111)
            temp_ax = ax
        fig.savefig(figname) 
    
    ## Cannot return ax object as argument
    del(arguments["ax"])
    # Return arguments that would recreate this graph
    return arguments
    


def line(
    x,
    y = [],
    ax=None,
    figname="",
    fig_kw = {},
    plot_kw =
        {
            "color": "tab:blue",
            "linewidth": 1,
        },
    xlabel_kw = 
        {
            "xlabel": "Targets",
            "fontsize": 16,
            "labelpad": 0,
        },
    ylabel_kw = 
        {
            "ylabel": "Predicted",
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
        }
    ):
    """
    Create a basic line plot using potentially multiple lines.
    
    Arguments:
    x: list/array
        Can be a single 
    
    """
    arguments = locals()
    arguments_copy = {}
    for key,value in arguments.items():
        if key == "x" or key == "y":
            arguments_copy[key] = list(value)
        elif key == "ax":
            arguments_copy[key] = value
        else:
            arguments_copy[key] = copy.deepcopy(value)
    arguments = arguments_copy
    
    fig = None
    if ax == None:
        fig = plt.figure(**arguments["fig_kw"])
        ax = fig.add_subplot(111)
        
    if len(arguments["y"]) > 0:
        ax.plot(x,y, **plot_kw)
    else:
        ax.plot(x, **plot_kw)
    
    labels_and_ticks(ax, 
                     arguments["xlabel_kw"], 
                     arguments["ylabel_kw"], 
                     arguments["xticks"], 
                     arguments["yticks"])
    
    ## Save if figname provided
    if len(figname) > 0:
        if fig == None:
            fig = plt.figure(**arguments["fig_kw"])
            temp_ax = fig.add_subplot(111)
            temp_ax = ax
        fig.savefig(figname)    
    
    ## Cannot return ax object as argument
    del(arguments["ax"])
    # Return arguments that would recreate this graph
    return arguments
    
    

def lines(
    values,
    ax=None,
    figname="",
    fig_kw = {},
    yaxis_assignment = [],
    legend_kw = 
        {
            "legend": True,
            "fontsize": 16,
        },
    colormap = 
        {
            "cmap": "",
            "truncate":
                {
                    "minval": 0.0,
                    "maxval": 1.0,
                    "n": 10000,
                }
        },
    plot_kw = 
        {
            "linewidth": 2,
            "colors": [],
            "labels": [],
        },   
    xlabel_kw = 
        {
            "xlabel": "",
            "fontsize": 16,
            "labelpad": 0,
        },
    ylabel_kw = 
        {
            "ylabel": "",
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
        },
    ):
    """
    Create a basic line plot composed of multiple lines.
    
    Arguments
    ---------
    values: list of lists/arrays
        Values for the lines you would like to plot. These must be formatted 
        such that each entry in the list has two lists/arrays, one for the 
        x values and one for the y values
    yaxis_assignment: list of int
        List of integers, either 0 or 1, that deteremines which y-axis the 
        given values will be plotted on. The 0 y-axis is the one on the 
        left-hand side of the plot. The 1 y-axis is on the right. This 
        second axis will be stored under ax.child_axes to be accesssible 
        outside of this function. 
    color_map: matplotlib.cm object
        Name of a colormap from matplotlib you would like to use. This is by
        far the easiest way to color multiple lines. If you would like more 
        control over coloring, use plot_kw["colors"]
    plot_kw: dict
        Keyword arguments fed into ax.plot function with some special keywords
        that are handled outside of ax.plot. These are:
            lists: Any keyword passed as a list is assumed to be used for the
                    iteration through the values arguments
        
        
    """
    arguments = locals()
    arguments_copy = {}
    for key,value in arguments.items():
        if key == "values":
            arguments_copy[key] = list(value)
        elif key == "ax":
            arguments_copy[key] = value
        else:
            arguments_copy[key] = copy.deepcopy(value)
    arguments = arguments_copy
    
    fig = None
    if ax == None:
        fig = plt.figure(**arguments["fig_kw"])
        ax = fig.add_subplot(111)
    
    ### Check for two y-axis
    use_right_yaxis = False
    if len(yaxis_assignment) > 0:
        ax1 = ax.twinx()
        if len(yaxis_assignment) != len(values):
            raise Exception("Argument yaxis_assignemt must be a list of "+
                        "0 or 1 integers for each value.")
        for entry in yaxis_assignment:
            if entry == 0:
                continue
            elif entry == 1:
                continue
            else:
                raise Exception("Argument yaxis_assignemt must be a list of "+
                        "0 or 1 integers for each value.")
        
        use_right_yaxis = True
        
        ### Add ax1 as a child axes so that it will still be accessible 
        ### outside of this function by the user
        ax.add_child_axes(ax1)
    
    ### Build color mapping for each line
    if len(colormap["cmap"]) == 0:
        if len(values) < 8:
            colormap["cmap"] = "Set1"
        else:
            colormap["cmap"] = "viridis"
            
    cmap = eval("cm.{}".format(colormap["cmap"]))
    cmap = truncate_colormap(cmap, **colormap["truncate"])
    color_list = colors_from_colormap(len(values), cmap)
    
    if "colors" in arguments["plot_kw"]:
        if len(arguments["plot_kw"]["colors"]) == 0:
            arguments["plot_kw"]["colors"] = color_list
    
    ### Colors was old keyword and has since been deprecated
    ### But I would not like to change behavior outside of this program 
    used_colors = False
    used_labels = False
    if "colors" in arguments["plot_kw"]:
        used_colors = True
        arguments["plot_kw"]["color"] = copy.deepcopy(arguments["plot_kw"]["colors"])
        del(arguments["plot_kw"]["colors"])
    if "labels" in arguments["plot_kw"]:
        used_labels = True
        arguments["plot_kw"]["label"] = copy.deepcopy(arguments["plot_kw"]["labels"])
        del(arguments["plot_kw"]["labels"])
    
    ### Should check over all arguments if an individual argument has been 
    ### provided for plot_kw
    multi_plot_kw_list = {}
    for key,value in arguments["plot_kw"].items():
        if type(value) == str:
            continue
        ### Check if iterable
        iterable = getattr(value, "__iter__", False)
        if callable(iterable):
            if len(value) > 0:
                multi_plot_kw_list[key] = value.copy()
    
    ## Temporarily delete to use only the rest of plot_kw in following loop
    ## and index over multi arguments
    for key in multi_plot_kw_list:
        del(arguments["plot_kw"][key])
    
    all_lines_list = []
    for idx,val in enumerate(values):
        temp_args = copy.deepcopy(arguments["plot_kw"])
        for key,value in multi_plot_kw_list.items():
            ## Check if there will be out of index issue. If there will be,  
            ## correct it by looping back around. 
            temp_idx = idx
            num_values = len(value)
            if idx >= num_values:
                loop = int(idx / num_values) + 1
                temp_idx = idx - loop*num_values
            
            temp_args[key] = value[temp_idx]
        
        ### Check for right yaxis
        if use_right_yaxis:
            temp_ax_list = [ax,ax1]
            temp_ax = temp_ax_list[yaxis_assignment[idx]]
        else:
            temp_ax = ax
            
        if len(val) == 1:
            all_lines_list.append(temp_ax.plot(val,**temp_args))
        elif len(val) == 2:
            all_lines_list.append(temp_ax.plot(val[0], val[1], **temp_args))
        else:
            raise Exception("Values passed into lines must be a list of "+
                    "lists of lists containing either one or two entries " +
                    "to plot as the x and y coordinates.")
    
    ### Need to put this here such that the xtick arguments are handled 
    ### Correctly
    if use_right_yaxis:
        label_and_ticks_right_yaxis(
            ax1,
            arguments["ylabel_kw_right"],
            arguments["yticks_right"])
        
    labels_and_ticks(ax, 
                     arguments["xlabel_kw"], 
                     arguments["ylabel_kw"], 
                     arguments["xticks"], 
                     arguments["yticks"])
    
    for key,value in multi_plot_kw_list.items():
        arguments["plot_kw"][key] = value
    
    if "labels" in arguments["plot_kw"]:
        temp_labels = arguments["plot_kw"]["labels"]
        if len(temp_labels) > 0 and arguments["legend_kw"]["legend"] == True:
            temp_legend_kw = arguments["legend_kw"].copy()
            del(temp_legend_kw["legend"])
            ax.legend(**temp_legend_kw)
            
    if "label" in arguments["plot_kw"]:
        temp_labels = arguments["plot_kw"]["label"]
        if len(temp_labels) > 0 and arguments["legend_kw"]["legend"] == True:
            temp_legend_kw = arguments["legend_kw"].copy()
            del(temp_legend_kw["legend"])
            
            combined_lines = all_lines_list[0]
            for entry in all_lines_list[1:]:
                combined_lines += entry
                
            ax.legend(combined_lines,
                      arguments["plot_kw"]["label"],
                      **temp_legend_kw)
    
    ## Cannot return ax object as argument
    del(arguments["ax"])
    
    ### Account for deprecated keyword but keep behavior the same
    if used_colors:
        arguments["plot_kw"]["colors"] = copy.deepcopy(arguments["plot_kw"]["color"])
        del(arguments["plot_kw"]["color"])
    if used_labels:
        arguments["plot_kw"]["labels"] = copy.deepcopy(arguments["plot_kw"]["label"])
        del(arguments["plot_kw"]["label"])
    
    # Return arguments that would recreate this graph
    return arguments
    
    

def hist(
    values,
    ax=None,
    figname="",
    fig_kw = {},
    hist_kw = 
        {
          "facecolor": "tab:blue",
          "edgecolor": "k"
        },   
    xlabel_kw = 
        {
            "xlabel": "",
            "fontsize": 16,
            "labelpad": 0,
        },
    ylabel_kw = 
        {
            "ylabel": "",
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
        }):
    """
    Create a basic histogram. 
    
    """
    arguments = locals()
    arguments_copy = {}
    for key,value in arguments.items():
        if key == "values":
            arguments_copy[key] = list(value)
        elif key == "ax":
            arguments_copy[key] = value
        else:
            arguments_copy[key] = copy.deepcopy(value)
    arguments = arguments_copy
    
    fig = None
    if ax == None:
        fig = plt.figure(**arguments["fig_kw"])
        ax = fig.add_subplot(111)
        
    ax.hist(values, **hist_kw)     
    
    labels_and_ticks(ax, 
                     arguments["xlabel_kw"], 
                     arguments["ylabel_kw"], 
                     arguments["xticks"], 
                     arguments["yticks"])
    
    ## Save if figname provided
    if len(figname) > 0:
        if fig == None:
            fig = plt.figure(**arguments["fig_kw"])
            temp_ax = fig.add_subplot(111)
            temp_ax = ax
        fig.savefig(figname) 
        
    
    ## Cannot return ax object as argument
    del(arguments["ax"])
    # Return arguments that would recreate this graph
    return arguments


def ahist(
    value_list,
    bins=10, 
    density=False,
    ax=None,
    figname="",
    fig_kw = {},
    bar_kw = 
        {
          "edgecolor": "k",
          "color_list": [],
          "alpha_list": [],
        },   
    colormap = 
        {
            "cmap": "viridis",
            "truncate":
                {
                    "minval": 0.0,
                    "maxval": 1.0,
                    "n": 10000,
                }
        },            
    xlabel_kw = 
        {
            "xlabel": "",
            "fontsize": 16,
            "labelpad": 0,
        },
    ylabel_kw = 
        {
            "ylabel": "",
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
        }): 
    """
    Prefered method for plotting multiple histograms on a single plot. The
    location of the bars for each plot are will be identical making cleaner
    looking comparisons of histograms.
    
    """
    arguments = locals()
    arguments_copy = {}
    for key,value in arguments.items():
        if key == "values":
            arguments_copy[key] = list(value)
        elif key == "ax":
            arguments_copy[key] = value
        else:
            arguments_copy[key] = copy.deepcopy(value)
    arguments = arguments_copy
    
    fig = None
    if ax == None:
        fig = plt.figure(**arguments["fig_kw"])
        ax = fig.add_subplot(111)
    
    ## Check for xlim to determine bin range behavior
    if len(arguments["xticks"]["xlim"]) != 0:
        pass
    else:
        ## Need to get min and max to have bins in these locations
        min_val = 0
        max_val = 0
        for values in value_list:
            temp_min = np.min(values)
            if temp_min < min_val:
                min_val = temp_min
            
            temp_max = np.max(values)
            if temp_max > max_val:
                max_val = temp_max
        
        arguments["xticks"]["xlim"] = [min_val, max_val]
    
    ## Set this at bottom so it's obvious how it's being used
    hist_range = arguments["xticks"]["xlim"]
    
    ## First compute the correct location of bin edges 
    bin_edges = np.linspace(hist_range[0], hist_range[1], bins + 1)
    centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
    heights = np.diff(bin_edges)
    
    ## Now bin the data
    binned_data_sets = [
        np.histogram(d, range=hist_range, bins=bins,
                     density=arguments["density"])[0]
        for d in value_list]
    
    ## Prepare colors and alpha
    cmap = eval("cm.{}".format(colormap["cmap"]))
    cmap = truncate_colormap(cmap, **colormap["truncate"])
    color_list = colors_from_colormap(len(value_list), cmap)
    
    if len(arguments["bar_kw"]["color_list"]) == 0:
        arguments["bar_kw"]["color_list"] = color_list
    
    if len(arguments["bar_kw"]["alpha_list"]) == 0:
        arguments["bar_kw"]["alpha_list"] = [1 for x in
                                             range(len(binned_data_sets))]
    
    ## Copy for use below. This will be directly fed into ax.bar() calls
    bar_kw = arguments["bar_kw"].copy()
    del(bar_kw["color_list"])
    del(bar_kw["alpha_list"])
    
    ## We are ready to complete plotting
    for i,binned_data in enumerate(binned_data_sets):
        
        ## Remove numerical noise
        idx = np.where(binned_data > 0.001)[0]
        temp_centers = centers[idx]
        binned_data = binned_data[idx]
        temp_height = heights[idx]
        
        color = arguments["bar_kw"]["color_list"][i]
        alpha = arguments["bar_kw"]["alpha_list"][i]
        
        ax.bar(temp_centers, binned_data, width=temp_height, 
               color=color,
               alpha=alpha,
               **bar_kw)
    
    labels_and_ticks(ax, 
                     arguments["xlabel_kw"], 
                     arguments["ylabel_kw"], 
                     arguments["xticks"], 
                     arguments["yticks"])
    
    ## Save if figname provided
    if len(figname) > 0:
        if fig == None:
            fig = plt.figure(**arguments["fig_kw"])
            temp_ax = fig.add_subplot(111)
            temp_ax = ax
        fig.savefig(figname) 
    
    ## Cannot return ax object as argument
    del(arguments["ax"])
    # Return arguments that would recreate this graph
    return arguments
    
    
def hline():
    """
    Puts horizontal line on graph. 
    
    """
    raise Exception("Not Implemented")
    

def vline():
    """
    Puts vertical line on graph. 
    
    """
    raise Exception("Not Implemented")
    
    
def hexhist():
    """
    
    """
    raise Exception("Not Implemented")
    
    
    
    
if __name__ == "__main__":
    pass
    
    
    