import matplotlib.pyplot as plt
# import colors
from matplotlib import cm
import numpy as np
from scipy import interpolate
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import numpy as np
import json
import os
from loguru import logger

renames = {
    "gpt-3.5-turbo": "chatgpt",
}

def color(model):
    model_sizes = [
        "ada",
        "babbage",
        "curie",
        "davinci",
        "turbo"
    ]
    
    base_map = cm.get_cmap("Blues")

    if "text" in model: 
        base_map = cm.get_cmap("Greens")
    if "003" in model or "turbo" in model: 
        base_map = cm.get_cmap("RdPu")
        
    for i, m in enumerate(model_sizes):
        if m in model:
            break
    
    return base_map(1 * (0.4 + (0.6 * (i / len(model_sizes)))))


def temperature_cmp(df, metric1, metric2, models=None, negate=False, clusters=True, xlim=None, ylim=None, cluster_label_positions=None, file=None, nolegend=False):
    if type(df) is str:
        df = pd.read_csv(df, sep="\t")

    d = df # df[df["temperature"] == 1.0]
    # d = d[np.abs(d["size"].values - 3000) < 20]
    try:
        d = d[d["few_shot"] == False]
        d = d[d["diverse"] == False]
    except Exception:
        pass
        
    # none
    conditions = []
    matchers = models if models else []
    for model_matcher in matchers:
        if model_matcher == "!text":
            model_matcher = "text"
            negate = True
        elif model_matcher == "text":
            model_matcher = "text"
            negate = False
        
        if model_matcher == "rlhf":
            matcher_condition = d["model"].str.contains("turbo") | d["model"].str.contains("003")
        else:
            matcher_condition = d["model"].str.contains(model_matcher)
            if model_matcher.endswith("$"):
                matcher_condition = d["model"].str.endswith(model_matcher[:-1])
            if model_matcher == "text":
                matcher_condition = matcher_condition & ~d["model"].str.contains("turbo") & ~d["model"].str.contains("003")
            if negate:
                matcher_condition = ~matcher_condition & ~d["model"].str.contains("turbo") & ~d["model"].str.contains("003")

        conditions = conditions + [matcher_condition]

    if len(conditions) > 0:
        d = d[np.logical_or.reduce(conditions)]

    models = list(set(d["model"].values))

    # sort models
    models = sorted(models) # , key=lambda x: x.replace("text-", ""))
    # move gpt-3.5-turbo to the end
    models = list(sorted(models, key=lambda x: x.replace("gpt-3.5-turbo", "zzzzzzz")))

    cluster_points = {}
    cluster_names = {
        "text": "Instruction-Tuned",
        "!text": "Vanilla",
        "rlhf": "RLHF"
    }
    cluster_colors = {
        # use different shades
        "text": color("text-davinci"),
        "!text": color("davinci"),
        "rlhf": color("gpt-3.5-turbo")
    }

    artists = {}

    for m in models:
        dm = d[d["model"] == m]
        # sort dm by temperature
        dm = dm.sort_values(by=["temperature"])
        # if len(dm) > 3:
        #     print(dm)
        metric1_values = dm[metric1]
        metric2_values = dm[metric2]
        label = renames.get(m, m)
        
        # smooth
        # sort x and y
        x,y = metric2_values.values, metric1_values.values
        # print(metric1_values, metric2_values)
        # three point spline from values
        fig = plt.plot(x, y, label=label, alpha=0.7, linewidth=2.5, color=color(m))
        
        # label first point with first temp
        # plt.annotate(dm["temperature"].values[0], (metric2_values.values[0], metric1_values.values[0]), color=color(m), fontsize=10)
        # add markers for all but last point
        plt.scatter(x[:-1], y[:-1], color=color(m), marker="o", s=10)
        
        # last point with last temp
        # plt.annotate(dm["temperature"].values[-1], (metric2_values.values[-1], metric1_values.values[-1]), color=color(m), fontsize=10)
        
        # add arrow to last temp
        xy = (metric2_values.values[-1], metric1_values.values[-1])
        xytext = (metric2_values.values[-2], metric1_values.values[-2])
        # normalize direction vector
        magnitude = np.sqrt((xy[0] - xytext[0])**2 + (xy[1] - xytext[1])**2)
        # normalize
        direction = ((xy[0] - xytext[0]) / magnitude, (xy[1] - xytext[1]) / magnitude)
        # scale
        l = 0.02
        xy = (xy[0] + direction[0] * l, xy[1] + direction[1] * l)
        xytext = (xy[0] - direction[0] * l, xy[1] - direction[1] * l)

        # # move along direction
        # xy = (xy[0] + 0.01, xy[1] + 0.01)

        plt.annotate("", xy=xy, xytext=xytext, arrowprops=dict(arrowstyle="->",color=color(m), linewidth=2.0, alpha=0.7), color=color(m))

        # if "turbo" in m or "003" in m:
        #     # annotate with model name at first point
        #     plt.annotate(label, (metric2_values.values[0] + 0.01, metric1_values.values[0]), color=color(m), fontsize=10)

        x_values = metric2_values.values
        y_values = metric1_values.values
        
        if "turbo" in m or "003" in m:
            cluster_points.setdefault("rlhf", []).append((x_values, y_values))
            # artists.setdefault("rlhf", []).append(fig[0])
        elif "text-" in m:
            cluster_points.setdefault("text", []).append((x_values, y_values))
            # artists.setdefault("text", []).append(fig[0])
        else:
            cluster_points.setdefault("!text", []).append((x_values, y_values))
        artists.setdefault("models", []).append(fig[0])

    if clusters:
        for c,points in cluster_points.items():
            # get the convex hull
            x_values = np.concatenate([x for x,y in points]).flatten()
            y_values = np.concatenate([y for x,y in points]).flatten()

            points = np.array(list(zip(x_values, y_values)))

            hull = ConvexHull(points)
            x_hull = np.append(points[hull.vertices,0],
                            points[hull.vertices,0][0])
            y_hull = np.append(points[hull.vertices,1],
                            points[hull.vertices,1][0])
            
            # # gaussian smooth
            # sigma = 0.4
            # x_hull = gaussian_filter1d(x_hull, sigma=sigma)
            # y_hull = gaussian_filter1d(y_hull, sigma=sigma)
            
            # interpolate
            dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
            dist_along = np.concatenate(([0], dist.cumsum()))
            spline, u = interpolate.splprep([x_hull, y_hull], 
                                            u=dist_along, s=0.0, per=1)
            interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
            interp_x, interp_y = interpolate.splev(interp_d, spline)

            col = cluster_colors[c]

            transparent_col = tuple(np.array(col) * 0.9)
            # plot with dotted boundary line, nofill
            plt.fill(interp_x, interp_y, alpha=0.1, color=transparent_col, linewidth=2.1, linestyle="dotted", zorder=0)
            # annotate at center
            darkgrey = col # tuple(np.array(col) * 0.95)
            # plt.annotate(cluster_names[c], (np.mean(interp_x), np.mean(interp_y)), color=darkgrey, fontsize=10)
            
            # add legend entry with marker as rectangle (add space to label)
            fig = plt.plot([], [], label=cluster_names[c], color=col, alpha=0.2, linewidth=2.5, marker="s", markersize=10, linestyle="None", markeredgecolor=darkgrey, markeredgewidth=1.5)

            artists.setdefault("clusters", []).append(fig[0])

    plt.ylabel(metric1.capitalize())
    plt.xlabel(metric2.capitalize())
    # no borders
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    # ticks
    plt.tick_params(axis='both', which='both', bottom=True, top=False, labelbottom=True, left=True, right=False, labelleft=True)

    # x and y label axis label font size and font family
    plt.gca().xaxis.label.set_size(18)
    plt.gca().yaxis.label.set_size(18)
    plt.gca().xaxis.label.set_fontname("Times New Roman")


    # remove y axis label
    plt.gca().yaxis.label.set_visible(False)
    # add custom y axis label
    plt.gca().annotate(metric1.capitalize(), xy=(-0.1, 0.5), xytext=(0.0, 1.07), xycoords='axes fraction', fontsize=18, ha='left', va='center', fontname="Times New Roman")

    # padding
    plt.gca().yaxis.labelpad = 40
    # padding top of entire canvas
    # grid
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
    if not nolegend:
        legends = []
        legend_artists = {k:v for k,v in artists.items() if k != "clusters"}
        for group, figs in legend_artists.items():
            figs += [plt.plot(np.zeros(1), np.zeros(1), color="w", alpha=0, label=" ")[0] for i in range(2)]

            l = plt.legend(handles=figs, fontsize=18, loc='lower left', 
                           bbox_to_anchor=(-0.3, -0.65), frameon=False,
                           ncol=3)
            # set font family in l
            for text in l.get_texts():
                text.set_fontname("Times New Roman")
            
    # set tick font to Times
    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontname("Times New Roman")
        label.set_fontsize(18)


    # add cluster labels in plot
    if clusters:
        # l = plt.legend(handles=artists["clusters"], fontsize=10, loc='upper right', bbox_to_anchor=(1.35, -0.02), frameon=False)
        for a in artists["clusters"]:
            # bounding box for Line2D object a
            bb = plt.gca().get_tightbbox(plt.gcf().canvas.get_renderer()).transformed(plt.gca().transData.inverted())
            
            # closer to black
            c = tuple(np.array(a.get_color() ) * 0.8)

            if cluster_label_positions is not None and a.get_label() in cluster_label_positions:
                offset = cluster_label_positions[a.get_label()]
            else:
                offset = (0,0)
            
            plt.annotate(a.get_label(), (bb.x0 + bb.width/2 + offset[0], bb.y0 + bb.height/2 + offset[1]), color=c, fontsize=16, ha="center", va="center", alpha=1.0, fontname="Times New Roman")

    # background color light grey
    plt.gca().set_facecolor('#f0f0f0')
    # no grid
    plt.grid(False)

    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)

    # legend position
    # plt.legend(loc='lower right', bbox_to_anchor=(1.45, -0.02), frameon=False)


    # plot dotted line at 0.5
    plt.plot([0.0, 1.0], [0.46, 0.46], color="grey", linestyle="dotted", linewidth=1.5, alpha=0.5, zorder=0)
    # add label to line Reference Dataset Diversity
    loc = (0.4, 0.4)
    plt.annotate("Reference\nDiversity", 
                 loc, color="grey", 
                 fontsize=14, ha="left", va="center", alpha=0.5, fontname="Times New Roman")


    if file is not None:
        plt.savefig(file, dpi=300, bbox_inches='tight')
    plt.close()



def dataset_cmp(df, metric1, metric2, models=None, negate=False, clusters=True, 
                xlim=None, ylim=None, cluster_label_positions=None, file=None, nolegend=False, profile = 1):
    if type(df) is str:
        df = pd.read_csv(df, sep="\t")

    d = df
    d = d[d["few_shot"] == False]
    d = d[d["diverse"] == False]
    d = d[d["top_p"] == 1.0]
    
    # none
    conditions = []
    matchers = models if models else []
    for model_matcher in matchers:
        if model_matcher == "!text":
            model_matcher = "text"
            negate = True
        elif model_matcher == "text":
            model_matcher = "text"
            negate = False
        
        if model_matcher == "rlhf":
            matcher_condition = d["model"].str.contains("turbo") | d["model"].str.contains("003")
        else:
            matcher_condition = d["model"].str.contains(model_matcher)
            if model_matcher.endswith("$"):
                matcher_condition = d["model"].str.endswith(model_matcher[:-1])
            if model_matcher == "text":
                matcher_condition = matcher_condition & ~d["model"].str.contains("turbo") & ~d["model"].str.contains("003")
            if negate:
                matcher_condition = ~matcher_condition & ~d["model"].str.contains("turbo") & ~d["model"].str.contains("003")

        conditions = conditions + [matcher_condition]

    # filter by condition
    if len(conditions) > 0:
        d = d[np.logical_or.reduce(conditions)]

    datasets = list(set(d["dataset"].values))

    # sort datasets
    datasets = sorted(datasets) # , key=lambda x: x.replace("text-", ""))
    # move gpt-3.5-turbo to the end
    datasets = list(sorted(datasets))

    cluster_points = {}
    cluster_names = {
        "text": "Instruction-Tuned",
        "!text": "Vanilla",
        "rlhf": "RLHF"
    }
    cluster_colors = {
        # use different shades
        "text": color("text-davinci"),
        "!text": color("davinci"),
        "rlhf": color("gpt-3.5-turbo")
    }

    artists = {}
    palette = cm.get_cmap("tab10")

    for ds in datasets:
        dm = d[d["dataset"] == ds]
        # sort dm by temperature
        dm = dm[dm["temperature"] == 1.0]
        # if len(dm) > 3:
        #     print(dm)
        metric1_values = dm[metric1]
        metric2_values = dm[metric2]
        label = renames.get(ds, ds)

        m = ds
        
        # smooth
        # sort x and y
        x,y = metric2_values.values, metric1_values.values
        # print(metric1_values, metric2_values)
        # three point spline from values
        fig = plt.plot(x, y, label=label, alpha=0.7, linewidth=2.5)
        
        model_scales = {
            "text-ada-001": "350M",
            "text-babbage-001": "1.2B",
            "text-curie-001": "6.7B",
            "text-davinci-001": "175B"
        }


        for i,m in enumerate(dm["model"].values):
            if i > 0 and i != len(dm["model"].values) - 1:
                continue
            offset = (0,0)
            
            if profile == 1:
                if ds == "SST" and m == "text-ada-001":
                    offset = (-0.015, -0.03)
                elif ds == "SST" and m == "text-davinci-001":
                    offset = (0.005, -0.03)
                if ds == "eli5" and m == "text-davinci-001":
                    offset = (0.005, 0.01)
                elif ds == "eli5" and m == "text-ada-001":
                    # up a bit
                    offset = (-0.015, 0.01)
                if ds == "AGNews" and m == "text-ada-001":
                    # right a bit
                    offset = (0.005, 0.005)
                elif ds == "AGNews" and m == "text-davinci-001":
                    # up a bit
                    offset = (-0.025, 0.015)
                if ds == "goemotions" and m == "text-davinci-001":
                    # up left a bit
                    offset = (-0.025, 0.02)
                elif ds == "goemotions" and m == "text-ada-001":
                    # up a bit
                    offset = (-0.025, 0.01)
            else:
                if ds == "SST" and m == "text-ada-001":
                    offset = (-0.015, -0.03)
                elif ds == "SST" and m == "text-davinci-001":
                    offset = (-0.025, 0.02)
                if ds == "eli5" and m == "text-davinci-001":
                    offset = (0.005, 0.01)
                elif ds == "eli5" and m == "text-ada-001":
                    # up a bit
                    offset = (-0.015, 0.01)
                if ds == "AGNews" and m == "text-ada-001":
                    # right a bit
                    offset = (-0.015,0.01)
                elif ds == "AGNews" and m == "text-davinci-001":
                    # up a bit
                    offset = (-0.01, -0.04)
                if ds == "goemotions" and m == "text-davinci-001":
                    # up left a bit
                    offset = (-0.01, -0.04)
                elif ds == "goemotions" and m == "text-ada-001":
                    # up a bit
                    offset = (-0.025, 0.01)
            plt.annotate(model_scales.get(m, m),
                        (metric2_values.values[i] + offset[0], metric1_values.values[i] + offset[1]), fontsize=10, 
                         color = fig[0].get_color())
        
        # # label first point with first model name
        # plt.annotate(model_scales.get(first_model, first_model),
        #             (metric2_values.values[0], metric1_values.values[0]), fontsize=10, 
        #              color = fig[0].get_color())
        
        # last point with last temp
        # plt.annotate(model_scales.get(last_model, last_model),
        #              (metric2_values.values[-1], metric1_values.values[-1]), fontsize=10,
        #              color= fig[0].get_color())
        
        # add markers for all but last point
        plt.scatter(x[:-1], y[:-1], marker="o", s=10, color=fig[0].get_color())
        
        # add arrow to last temp
        xy = (metric2_values.values[-1], metric1_values.values[-1])
        xytext = (metric2_values.values[-2], metric1_values.values[-2])
        # normalize direction vector
        magnitude = np.sqrt((xy[0] - xytext[0])**2 + (xy[1] - xytext[1])**2)
        # normalize
        direction = ((xy[0] - xytext[0]) / magnitude, (xy[1] - xytext[1]) / magnitude)
        # scale
        l = 0.02
        xy = (xy[0] + direction[0] * l, xy[1] + direction[1] * l)
        xytext = (xy[0] - direction[0] * l, xy[1] - direction[1] * l)

        # # move along direction
        # xy = (xy[0] + 0.01, xy[1] + 0.01)

        plt.annotate("", xy=xy, xytext=xytext, 
                     arrowprops=dict(arrowstyle="->",color=fig[0].get_color(), linewidth=2.0, alpha=0.7), 
                     color=fig[0].get_color(), fontsize=10)

        # if "turbo" in m or "003" in m:
        #     # annotate with model name at first point
        #     plt.annotate(label, (metric2_values.values[0] + 0.01, metric1_values.values[0]), color=color(m), fontsize=10)

        x_values = metric2_values.values
        y_values = metric1_values.values
        
        if "turbo" in m or "003" in m:
            cluster_points.setdefault("rlhf", []).append((x_values, y_values))
            # artists.setdefault("rlhf", []).append(fig[0])
        elif "text-" in m:
            cluster_points.setdefault("text", []).append((x_values, y_values))
            # artists.setdefault("text", []).append(fig[0])
        else:
            cluster_points.setdefault("!text", []).append((x_values, y_values))
        artists.setdefault("models", []).append(fig[0])

    if clusters:
        for c,points in cluster_points.items():
            # get the convex hull
            x_values = np.concatenate([x for x,y in points]).flatten()
            y_values = np.concatenate([y for x,y in points]).flatten()

            points = np.array(list(zip(x_values, y_values)))

            hull = ConvexHull(points)
            x_hull = np.append(points[hull.vertices,0],
                            points[hull.vertices,0][0])
            y_hull = np.append(points[hull.vertices,1],
                            points[hull.vertices,1][0])
            
            # interpolate
            dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
            dist_along = np.concatenate(([0], dist.cumsum()))
            spline, u = interpolate.splprep([x_hull, y_hull], 
                                            u=dist_along, s=0.0, per=1)
            interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
            interp_x, interp_y = interpolate.splev(interp_d, spline)

            col = cluster_colors[c]

            transparent_col = tuple(np.array(col) * 0.9)
            # plot with dotted boundary line, nofill
            plt.fill(interp_x, interp_y, alpha=0.1, color=transparent_col, linewidth=2.1, linestyle="dotted", zorder=0)
            # annotate at center
            darkgrey = col # tuple(np.array(col) * 0.95)
            # plt.annotate(cluster_names[c], (np.mean(interp_x), np.mean(interp_y)), color=darkgrey, fontsize=10)
            
            # add legend entry with marker as rectangle (add space to label)
            fig = plt.plot([], [], label=cluster_names[c], color=col, alpha=0.2, linewidth=2.5, marker="s", markersize=10, linestyle="None", markeredgecolor=darkgrey, markeredgewidth=1.5)

            artists.setdefault("clusters", []).append(fig[0])

    plt.ylabel(metric1.capitalize())
    plt.xlabel(metric2.capitalize())
    # no borders
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    # ticks
    plt.tick_params(axis='both', which='both', bottom=True, top=False, labelbottom=True, left=True, right=False, labelleft=True)

    # x and y label axis label font size and font family
    plt.gca().xaxis.label.set_size(18)
    plt.gca().yaxis.label.set_size(18)
    plt.gca().xaxis.label.set_fontname("Times New Roman")


    # remove y axis label
    plt.gca().yaxis.label.set_visible(False)
    # add custom y axis label
    plt.gca().annotate(metric1.capitalize(), xy=(-0.1, 0.5), xytext=(0.0, 1.07), xycoords='axes fraction', fontsize=18, ha='left', va='center', fontname="Times New Roman")

    # padding
    plt.gca().yaxis.labelpad = 40
    # padding top of entire canvas
    # grid
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
    if not nolegend:
        legends = []
        legend_artists = {k:v for k,v in artists.items() if k != "clusters"}
        for group, figs in legend_artists.items():
            figs += [plt.plot(np.zeros(1), np.zeros(1), color="w", alpha=0, label=" ")[0] for i in range(2)]

            l = plt.legend(handles=figs, fontsize=18, loc='lower left', frameon=False, ncol=3)
            # set font family in l
            for text in l.get_texts():
                text.set_fontname("Times New Roman")
            
    # set tick font to Times
    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontname("Times New Roman")
        label.set_fontsize(18)


    # add cluster labels in plot
    if clusters:
        # l = plt.legend(handles=artists["clusters"], fontsize=10, loc='upper right', bbox_to_anchor=(1.35, -0.02), frameon=False)
        for a in artists["clusters"]:
            # bounding box for Line2D object a
            bb = plt.gca().get_tightbbox(plt.gcf().canvas.get_renderer()).transformed(plt.gca().transData.inverted())
            
            # closer to black
            c = tuple(np.array(a.get_color() ) * 0.8)

            if cluster_label_positions is not None and a.get_label() in cluster_label_positions:
                offset = cluster_label_positions[a.get_label()]
            else:
                offset = (0,0)
            
            plt.annotate(a.get_label(), (bb.x0 + bb.width/2 + offset[0], bb.y0 + bb.height/2 + offset[1]), color=c, fontsize=16, ha="center", va="center", alpha=1.0, fontname="Times New Roman")

    # background color light grey
    plt.gca().set_facecolor('#f0f0f0')
    # no grid
    plt.grid(False)

    plt.xlim(xlim)
    plt.ylim(ylim)

    # legend position
    # plt.legend(loc='lower right', bbox_to_anchor=(1.45, -0.02), frameon=False)

    if file is not None:
        plt.savefig(file, dpi=300, bbox_inches='tight')
    plt.close()


def metric_cmp_datasets(df, metric1, negate=False, clusters=True, xlim=None, ylim=None, include_real=False, save_file=None, add_label=False):
    if type(df) is str:
        df = pd.read_csv(df, sep="\t")

    d = df 

    model_order1 = [
        "ada", 
        "babbage",
        "curie", 
        "davinci"
    ]

    model_order2 = [
        "text-ada-001",
        "text-babbage-001",
        "text-curie-001",
        "text-davinci-001"
    ]

    model_order3 = [
        "text-davinci-003",
        "z"
    ]
    renames = {"gpt-3.5-turbo": "z"}
    d["model"] = d["model"].apply(lambda x: renames[x] if x in renames else x)

    d = d.sort_values(by=["model"])

    real_metric = d[d["model"] == "real"][metric1].values[0]

    fig, ax = plt.subplots()

    dm = d[d["model"].isin(model_order1)]
    metric1_values = np.array(dm[metric1])
    ax.plot(np.arange(0, len(metric1_values), 1), metric1_values, marker="o", label="Vanilla", alpha=0.7, linewidth=2.5, color=cm.get_cmap("Blues")(0.7))

    dm = d[d["model"].isin(model_order2)]
    metric2_values = np.array(dm[metric1])
    ax.plot(np.arange(0, len(metric2_values), 1), metric2_values, marker="o", label="Instruction-Tuned", alpha=0.7, linewidth=2.5, color=cm.get_cmap("Greens")(0.7))

    dm = d[d["model"].isin(model_order3)]
    metric2_values = np.array(dm[metric1])
    ax.plot(np.arange(len(model_order2), len(model_order2) + len(metric2_values), 1), metric2_values, label="RLHF", marker="o", alpha=0.7, linewidth=2.5, color=cm.get_cmap("RdPu")(0.7))

    ax.plot([0, 5], [real_metric] * 2, label="Reference", linestyle='--', alpha=0.7, linewidth=2.5, color=cm.get_cmap("Greys")(0.7))


    plt.gca().xaxis.label.set_size(16)
    plt.gca().yaxis.label.set_size(16)
    plt.gca().xaxis.label.set_fontname("Times New Roman")

    # remove y axis label
    plt.gca().yaxis.label.set_visible(False)
    # add custom y axis label
    plt.gca().annotate(metric1.capitalize(), xy=(-0.1, 0.5), xytext=(0.21, 1.07), xycoords='axes fraction', fontsize=22, ha='right', va='center', fontname="Times New Roman")
    ax.set_xlabel("Model Size/Type")
    # no borders
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # hide other borders too, only ticks
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    # padding
    plt.gca().yaxis.labelpad = 40

    model_labels = ["350M", "1.2B", "6.7B", "175B", "PPO-175B", "Chat-175B"]
    ax.set_xticks(np.arange(0, len(model_labels), 1))
    ax.set_xticklabels(model_labels)
    # ticks
    plt.tick_params(axis='both', which='both', bottom=True, top=False, labelbottom=True, left=True, right=False, labelleft=True)
    # grid
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    # background color light grey
    plt.gca().set_facecolor('#f0f0f0')
    # no grid
    plt.grid(False)

    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)

    # legend position
    if add_label:
        ax.legend(loc='lower right', bbox_to_anchor=(1.45, -0.02), frameon=False)
        legend = ax.legend(loc='lower right', bbox_to_anchor=(1.45, -0.02), frameon=False)

        def export_legend(legend, filename="../plots/metrics/legend.pdf"):
            fig  = legend.figure
            fig.canvas.draw()
            bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(filename, dpi="figure", bbox_inches=bbox)
        export_legend(legend)

    if save_file is not None:
        fig.savefig(save_file)

    plt.close()


def get_results(folder, include_std=False):
    pipelines = [json.load(open(os.path.join(folder, "pipelines", f"pipeline_{file}.json"), "r")) 
                 for file in range(len(os.listdir(os.path.join(folder, "pipelines"))))]
    results = None
    for file in os.listdir(os.path.join(folder, "results")):
        if file.startswith("list_dict_metrics"):
            results = json.load(open(os.path.join(folder, "results", file), "r"))
            break
    
    results.append(json.load(open(os.path.join(folder, "results", "dict_real_metrics.json"), "r")))

    dataframe = pd.DataFrame(results)
    columns = {
        "complexity": "finetune_temporal_fake_to_fake.accuracy", 
        "diversity": "average_distinctness_spacy.1",
        "performance": "finetune_temporal_fake_to_real.accuracy", 
        "faithfulness": "finetune_temporal_real_to_fake.accuracy", 
        "conformity": "mauve"
    }

    if include_std:
        cols = list(columns.keys())
        for metric in cols:
            columns[metric + "_std"] = columns[metric] + "_std" 

    generators = [pipeline["generator"] if pipeline["generator"]["class"] == "Generator" else pipeline["generator"]["converters"][0] for pipeline in pipelines]
    results = dataframe[list(columns.values())].reset_index(drop=True)
    results.columns = list(columns.keys())
    results["temperature"] = [generator["querier"]["temperature"] for generator in generators][:len(results) - 1] + [1.0]
    results["model"] = [generator["querier"]["model"] for generator in generators][:len(results) - 1] + ["real"]
    results["top_p"] = [generator["querier"].get("top_p", None) for generator in generators][:len(results) - 1] + [1.0]
    classes = list(pipelines[0]["generator"]["prompts"]["prompts"].keys())
    results["few_shot"] = [len(generator["prompts"]["prompts"][classes[0]]) > 5 for generator in generators][:len(results) - 1] + [False]
    results["diverse"] = [pipeline["generator"]["class"] != "Generator" for pipeline in pipelines][:len(results) - 1] + [False]
    results["diverse_bad"] = [generator["querier"].get("presence_penalty", 0) > 0 for generator in generators][:len(results)] + [False]

    results["complexity"] = 1 - results["complexity"]

    return results

def get_results_eval(eval_folder, include_std=False):
    results = pd.DataFrame()
    for subfolder in os.listdir(eval_folder):
        new_results = get_results(os.path.join(eval_folder, subfolder), include_std=include_std)
        results = pd.concat([results, new_results], ignore_index=True)
    
    return results



if __name__ == "__main__":
    datasets = ["AGNews", "eli5", "goemotions", "SST"]
    datasets_real = []
    os.makedirs("../processed", exist_ok=True)
    for dataset in datasets:
        results = get_results_eval(f"../data/generations/{dataset}/meta_eval")
        results.fillna(1, inplace=True)
        results.to_csv(f"../processed/{dataset}.csv", sep="\t", index=False)
        datasets_real.append(dataset)

    all = pd.DataFrame()
    for dataset in datasets_real:
        df = pd.read_csv(f"../processed/{dataset}.csv", sep="\t")
        df["dataset"] = dataset
        all = all.append(df)

    normal = all[np.logical_not(all["few_shot"])]
    normal = normal[np.logical_not(normal["diverse"])]
    normal = normal[normal["top_p"] == 1.0]
    normal = normal[np.logical_not(normal["diverse_bad"])]
    normal_all = normal.append(all[all["model"].str.contains("real")])
    aggregated = normal_all.groupby(["model", "temperature"]).mean().reset_index()
    os.makedirs("../plots/metrics", exist_ok=True)
    for metric in ["complexity", "diversity", "faithfulness", "performance", "conformity"]:
        metric_cmp_datasets(aggregated[aggregated["temperature"] == 1], metric, ylim=(0,1), add_label=False, save_file=f"../plots/metrics/{metric}.pdf")

    for dataset in datasets_real:
        os.makedirs(f"../plots/metrics/{dataset}", exist_ok=True)
        normal_all_data = normal_all[normal_all["dataset"] == dataset]
        aggregated_data = normal_all_data.groupby(["model", "temperature"]).mean().reset_index()
        for metric in ["complexity", "diversity", "faithfulness", "performance", "conformity"]:
            metric_cmp_datasets(aggregated_data[aggregated_data["temperature"] == 1], metric, ylim=(0,1), add_label=False, 
                                save_file=f"../plots/metrics/{dataset}/{metric}.pdf")
   
    metric_cmp_datasets(aggregated[aggregated["temperature"] == 1], metric, ylim=(0,1), add_label=True)

    temperature_cmp(aggregated[aggregated["model"] != "real"], "diversity", "faithfulness", clusters=True, models=[], cluster_label_positions={
        "Vanilla": (-0.13, 0.15),
        "RLHF": (0.17, -0.235),
        "Instruction-Tuned": (0.15, 0.22),
    }, file="../plots/div-faith.pdf", nolegend=True, xlim=(0.4,1.0), ylim=(0.1, 0.7))

    temperature_cmp(aggregated[aggregated["model"] != "real"], "diversity", "conformity", clusters=True, models=[], cluster_label_positions={
        "Vanilla": (-0.15, 0.15),
        "RLHF": (-0.1, -0.21),
        "Instruction-Tuned": (0.14, 0.21),
    }, file="../plots/div-conf.pdf", nolegend=True, xlim=(0.0,0.6), ylim=(0.1,0.7))

    temp1 = normal_all[normal_all["temperature"] == 1.0]
    temp1 = temp1[temp1["model"].str.startswith("text")]
    temp1 = temp1[temp1["model"] != "text-davinci-003"]
    temp1 = temp1.sort_values(by=["model"])
    dataset_cmp(temp1, "diversity", "faithfulness", models=[], clusters=False, file="../plots/div-faith-datasets.pdf", 
                xlim=(0.6,1.0), ylim=(0,0.6))
    dataset_cmp(temp1, "performance", "conformity", models=[], clusters=False, file="../plots/perf-conf-datasets.pdf",
                xlim=(0,0.5), ylim=(0.4,0.9))
