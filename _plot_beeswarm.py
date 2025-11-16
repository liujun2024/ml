import matplotlib.pyplot as pl
from ._colors import red_blue, blue_rgb
import numpy as np


# TODO: remove unused title argument / use title argument
# TODO: Add support for hclustering based explanations where we sort the leaf order by magnitude and then show the dendrogram to the left
def summary_legacy(shap_values, features=None, feature_names=None, max_display=None, plot_type=None,
                 color=None, axis_color="#333333", title=None, alpha=1, show=True, sort=True,
                 color_bar=True, plot_size="auto", layered_violin_max_num_bins=20, class_names=None,
                 class_inds=None,
                 color_bar_label="Normalized feature value",
                 cmap=red_blue,
                 show_values_in_legend=False,
                 # depreciated
                 auto_size_plot=None,
                 use_log_scale=False):
    """Create a SHAP beeswarm plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : numpy.array
        For single output explanations this is a matrix of SHAP values (# samples x # features).
        For multi-output explanations this is a list of such matrices of SHAP values.

    features : numpy.array or pandas.DataFrame or list
        Matrix of feature values (# samples x # features) or a feature_names list as shorthand

    feature_names : list
        Names of the features (length # features)

    max_display : int
        How many top features to include in the plot (default is 20, or 7 for interaction plots)

    plot_type : "dot" (default for single output), "bar" (default for multi-output), "violin",
        or "compact_dot".
        What type of summary plot to produce. Note that "compact_dot" is only used for
        SHAP interaction values.

    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default the size is auto-scaled based on the number of
        features that are being displayed. Passing a single float will cause each row to be that
        many inches high. Passing a pair of floats will scale the plot by that
        number of inches. If None is passed then the size of the current figure will be left
        unchanged.

    show_values_in_legend: bool
        Flag to print the mean of the SHAP values in the multi-output bar plot. Set to False
        by default.
    """

    multi_class = False

    # assert len(shap_values.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."

    plot_type = "dot" # default for single output explanations
    
    # default color:
    color = blue_rgb

    idx2cat = None
    # # convert from a DataFrame or other types
    # if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
    #     if feature_names is None:
    #         feature_names = features.columns
    #     # feature index to category flag
    #     idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
    #     features = features.values
    # elif isinstance(features, list):
    #     if feature_names is None:
    #         feature_names = features
    #     features = None
    # elif (features is not None) and len(features.shape) == 1 and feature_names is None:
    #     feature_names = features
    #     features = None

    num_features = (shap_values[0].shape[1] if multi_class else shap_values.shape[1])

    if features is not None:
        shape_msg = "The shape of the shap_values matrix does not match the shape of the " \
                    "provided data matrix."
        if num_features - 1 == features.shape[1]:
            assert False, shape_msg + " Perhaps the extra column in the shap_values matrix is the " \
                          "constant offset? Of so just pass shap_values[:,:-1]."
        else:
            assert num_features == features.shape[1], shape_msg

    if max_display is None:
        # max_display = 5
        max_display = 20

    if sort:
        # order features by the sum of their effect magnitudes
        if multi_class:
            feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
        else:
            feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_order = feature_order[-min(max_display, len(feature_order)):]
    # else:
    # feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

    row_height = 0.4
    # if plot_size == "auto":
        # pl.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    
    # elif type(plot_size) in (list, tuple):
    #     pl.gcf().set_size_inches(plot_size[0], plot_size[1])
    # elif plot_size is not None:
    #     pl.gcf().set_size_inches(8, len(feature_order) * plot_size + 1.5)
    # pl.axvline(x=0, color="#999999", zorder=-1)
    pl.axvline(x=0, color="silver", zorder=-1, ls='--', lw=1.2)

    if plot_type == "dot":
        for pos, i in enumerate(feature_order):
            # pl.axhline(y=pos, color="#cccccc", lw=0.5, ls='--', zorder=-1)
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(4, 2), zorder=-1)
            shaps = shap_values[:, i]
            values = None if features is None else features[:, i]
            inds = np.arange(len(shaps))
            np.random.shuffle(inds)
            if values is not None:
                values = values[inds]
            shaps = shaps[inds]
            colored_feature = True
            try:
                values = np.array(values, dtype=np.float64)  # make sure this can be numeric
            except Exception:
                colored_feature = False
            N = len(shaps)
            # hspacing = (np.max(shaps) - np.min(shaps)) / 200
            # curr_bin = []
            nbins = 100
            quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
            inds = np.argsort(quant + np.random.randn(N) * 1e-6)
            layer = 0
            last_bin = -1
            ys = np.zeros(N)
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]
            ys *= 0.9 * (row_height / np.max(ys + 1))

            if features is not None and colored_feature:
                # trim the color range, but prevent the color range from collapsing
                vmin = np.nanpercentile(values, 5)
                vmax = np.nanpercentile(values, 95)
                if vmin == vmax:
                    vmin = np.nanpercentile(values, 1)
                    vmax = np.nanpercentile(values, 99)
                    if vmin == vmax:
                        vmin = np.min(values)
                        vmax = np.max(values)
                if vmin > vmax: # fixes rare numerical precision issues
                    vmin = vmax

                assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

                # plot the nan values in the interaction feature as grey
                nan_mask = np.isnan(values)
                pl.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777",
                           s=16, alpha=alpha, linewidth=0.05,
                           zorder=3, rasterized=len(shaps) > 500,
                           marker='$\u25EF$',
                           )

                # plot the non-nan values colored by the trimmed feature value
                cvals = values[np.invert(nan_mask)].astype(np.float64)
                cvals_imp = cvals.copy()
                cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
                cvals[cvals_imp > vmax] = vmax
                cvals[cvals_imp < vmin] = vmin
                pl.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                           cmap=cmap, vmin=vmin, vmax=vmax, s=16,
                           c=cvals, alpha=alpha, linewidth=0.05,
                           zorder=3, rasterized=len(shaps) > 500,
                           marker='$\u25EF$',
                           )
            else:

                pl.scatter(shaps, pos + ys, s=16, alpha=alpha, linewidth=0.05, zorder=3,
                           color=color if colored_feature else "#777777", rasterized=len(shaps) > 500,
                           marker='$\u25EF$',
                           )

    # draw the color bar
#     if color_bar and features is not None and plot_type != "bar" and \
#             (plot_type != "layered_violin" or color in pl.cm.datad):
#         import matplotlib.cm as cm
#         m = cm.ScalarMappable(cmap=cmap if plot_type != "layered_violin" else pl.get_cmap(color))
#         m.set_array([0, 1])
#         cb = pl.colorbar(m, ax=pl.gca(), ticks=[0, 1], aspect=80)
#         # cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
#         cb.set_label(color_bar_label, size=12, labelpad=0)
#         cb.ax.tick_params(labelsize=11, length=0)
#         cb.set_alpha(1)
#         cb.outline.set_visible(False)
# #         bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
# #         cb.ax.set_aspect((bbox.height - 0.9) * 20)
#         # cb.draw_all()

    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    # pl.gca().spines['right'].set_visible(False)
    # pl.gca().spines['top'].set_visible(False)
    # pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    # pl.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=13)
    if plot_type != "bar":
        pl.gca().tick_params('y', length=20, width=0.5, which='major')
    # pl.gca().tick_params('x', labelsize=11)
    pl.ylim(-1, len(feature_order))
    # if plot_type == "bar":
    #     pl.xlabel(labels['GLOBAL_VALUE'], fontsize=13)
    # else:
    #     pl.xlabel(labels['VALUE'], fontsize=13)
    # pl.tight_layout()
    return pl.gca()

    if show:
        pl.show()


if __name__ == "__main__":
    pass
