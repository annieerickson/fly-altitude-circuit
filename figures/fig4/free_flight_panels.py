"""Free-flight plots"""
from fly_altitude_circuit import config, plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def plot_traces(results, panel):
    style = config.load('style')
    line_width_spines = style['line']['spines']
    tick_width = style['line']['tick_width']
    tick_length = style['line']['tick_length']
    axes_title_font = style['font']['axes_title']
    ticklabel_font = style['font']['ticklabel']
    driverlabel_font = style['font']['ticklabel']
    drivercolor = style['free_flight']['experiment_color']
    hexcode_630nm = style['free_flight']['illumination_color']
    line_width = panel['line_width']
    axis_ticklabel_fontsize = panel['axis_ticklabel_fontsize']
    axis_label_fontsize = panel['axis_label_fontsize']
    fig_width = panel['fig_width']
    fig_height = panel['fig_height']
    time = panel['time']
    ylims, yticks = panel['ylim'], panel['yticks']
    xlims, xticks = panel['xlim'], panel['xticks']
    y_axis_txt = panel['ylabel']

    alpha_CI, alpha_patch = panel['alpha_CI'], panel['alpha_patch']
    df_exp_real_mean_CI = results["traces"]['df_exp_real_mean_CI']
    df_exp_sham_mean_CI = results["traces"]['df_exp_sham_mean_CI']
    df_control_real_mean_CI = results["traces"]['df_control_real_mean_CI']
    df_control_sham_mean_CI = results["traces"]['df_control_sham_mean_CI']
    fig, ax = plt.subplots(1, 2)
    sham_axis = 1
    light_axis = 0
    ax[light_axis].plot(df_exp_real_mean_CI[0], df_exp_real_mean_CI[1], color=drivercolor, linestyle='-', linewidth=line_width)
    ax[light_axis].fill_between(df_exp_real_mean_CI[0], df_exp_real_mean_CI[3], df_exp_real_mean_CI[2], color=drivercolor, alpha=alpha_CI, edgecolor='none')
    ax[light_axis].plot(df_control_real_mean_CI[0], df_control_real_mean_CI[1], color=style['free_flight']['control_color'], linestyle='-', linewidth=line_width)
    ax[light_axis].fill_between(df_control_real_mean_CI[0], df_control_real_mean_CI[3], df_control_real_mean_CI[2], color=style['free_flight']['control_color'], alpha=alpha_CI, edgecolor='none')
    ax[sham_axis].plot(df_exp_sham_mean_CI[0], df_exp_sham_mean_CI[1], color=drivercolor, linestyle='-', linewidth=line_width)
    ax[sham_axis].fill_between(df_exp_sham_mean_CI[0], df_exp_sham_mean_CI[3], df_exp_sham_mean_CI[2], color=drivercolor, alpha=alpha_CI, edgecolor='none')
    ax[sham_axis].plot(df_control_sham_mean_CI[0], df_control_sham_mean_CI[1], color=style['free_flight']['control_color'], linestyle='-', linewidth=line_width)
    ax[sham_axis].fill_between(df_control_sham_mean_CI[0], df_control_sham_mean_CI[3], df_control_sham_mean_CI[2], color=style['free_flight']['control_color'], alpha=alpha_CI, edgecolor='none')
    light_on = np.arange(675) / 1000
    ax[light_axis].fill_between(light_on, ylims[0], ylims[-1], color=hexcode_630nm, alpha=alpha_patch, linewidth=0.0001)
    ax[sham_axis].fill_between(light_on, ylims[0] + 0.005, ylims[-1], edgecolor=hexcode_630nm, facecolor='w', alpha=alpha_patch, linewidth=line_width)
    ax[0].set_ylabel(y_axis_txt, fontsize=axis_label_fontsize)
    ax[0].set_xlabel(f'time ({time})', fontsize=axis_label_fontsize, labelpad=-1)
    ax[1].set_xlabel(f'time ({time})', fontsize=axis_label_fontsize, labelpad=-1)
    for i in range(2):
        ax[i].set_ylim(ylims)
        ax[i].set_yticks(yticks)
        ax[i].set_yticklabels(ax[i].get_yticks(), fontsize=axis_ticklabel_fontsize)
        ax[i].set_xlim(xlims)
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels(ax[i].get_xticks(), fontsize=axis_ticklabel_fontsize)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines.bottom.set_bounds((xticks[0], xticks[-1]))
        ax[i].spines.left.set_bounds((yticks[0], yticks[-1]))
        ax[i].yaxis.set_tick_params(which='major', direction='in', width=tick_width, length=tick_length)
        ax[i].xaxis.set_tick_params(which='major', direction='in', width=tick_width, length=tick_length)
        ax[i].spines.left.set_linewidth(line_width_spines)
        ax[i].spines.bottom.set_linewidth(line_width_spines)
    ax[light_axis].set_title('Light on', fontsize=axes_title_font)
    ax[sham_axis].set_title('Light off', fontsize=axes_title_font)
    fig.set_size_inches(fig_width, fig_height)
    fig.subplots_adjust(**panel['layout'])
    return fig

def plot_statistics(results, panel):
    style = config.load('style')
    line_width_spines = style['line']['spines']
    tick_width = style['line']['tick_width']
    tick_length = style['line']['tick_length']
    axes_title_font = style['font']['axes_title']
    ticklabel_font = style['font']['ticklabel']
    driverlabel_font = style['font']['ticklabel']
    drivercolor = style['free_flight']['experiment_color']
    hexcode_630nm = style['free_flight']['illumination_color']
    line_width = panel['line_width']
    axis_ticklabel_fontsize = panel['axis_ticklabel_fontsize']
    axis_label_fontsize = panel['axis_label_fontsize']
    fig_width = panel['fig_width']
    fig_height = panel['fig_height']
    time = panel['time']
    ylims, yticks = panel['ylim'], panel['yticks']
    xlims, xticks = panel['xlim'], panel['xticks']
    y_axis_txt = panel['ylabel']

    scalar_groups = results['scalar_groups']
    fig, axs = plt.subplots(1, 1)
    name_plot_list = ['ControlxSham', 'CsChrimsonxSham', 'ControlxLightOn', 'CsChrimsonxLightOn']
    for i, data_dict in enumerate(scalar_groups):
        grand_mean = data_dict['mean_change']
        data_points = data_dict['altitude_change_data']
        CI_list = data_dict['CI']
        jitter = data_dict['jitter']
        axs.scatter(np.full_like(data_points, i) + jitter, data_points, color=style['free_flight']['point_color'], alpha=style['free_flight']['point_alpha'], s=style['free_flight']['point_size'], zorder=3, edgecolor='none')
        ci_height = CI_list[1] - CI_list[0]
        rect = Rectangle((i - style['free_flight']['interval_width'] / 2, CI_list[0]), width=style['free_flight']['interval_width'], height=ci_height, facecolor=style['free_flight']['interval_color'], alpha=style['free_flight']['interval_alpha'], edgecolor='none', zorder=1)
        axs.add_patch(rect)
        axs.hlines(grand_mean, i - style['free_flight']['interval_width'] / 2, i + style['free_flight']['interval_width'] / 2, color=style['free_flight']['mean_color'], lw=line_width, zorder=2)
    axs.set_ylim(ylims)
    axs.set_yticks(yticks)
    axs.spines.left.set_bounds((yticks[0], yticks[-1]))
    axs.set_xlim(xlims)
    axs.set_xticks(xticks)
    axs.spines.bottom.set_bounds((xticks[0], xticks[-1]))
    axs.set_ylabel(y_axis_txt, fontsize=axes_title_font, rotation=90, labelpad=5, va='center')
    axs.yaxis.set_tick_params(which='major', direction='in')
    axs.xaxis.set_tick_params(which='major', direction='in')
    axs.spines.left.set_linewidth(line_width_spines)
    axs.spines.bottom.set_linewidth(line_width_spines)
    axs.tick_params(axis='x', which='major', labelsize=driverlabel_font, width=tick_width, length=tick_length)
    axs.tick_params(axis='y', which='major', labelsize=ticklabel_font, width=tick_width, length=tick_length)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.set_xticks(range(len(name_plot_list)))
    axs.set_xticklabels(name_plot_list, fontsize=ticklabel_font, rotation=45)
    fig.set_size_inches(fig_width, fig_height)
    fig.subplots_adjust(**panel['layout'])
    plotting.add_canvas_padding(fig, panel.get('canvas_padding_in'))
    return fig

