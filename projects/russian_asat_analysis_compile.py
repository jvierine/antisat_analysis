#!/usr/bin/env python
# coding: utf-8

# # Compile analysis
# 
# first we list all the relevant result files

# In[1]:


from astropy.time import Time, TimeDelta
import h5py
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# plt.style.use('dark_background')

import sorts
import pyorb
import pyant


# In[2]:


HERE = Path(__file__).parent.resolve()
OUTPUT = HERE / 'output' / 'russian_asat'
out_plots = OUTPUT / 'categorization'
out_plots.mkdir(exist_ok=True)
print(f'Using {OUTPUT} as output')

clobber = False
# fig_format = 'eps'
fig_format = 'png'

# In[3]:


with open(OUTPUT / 'paths.pickle', 'rb') as fh:
    paths = pickle.load(fh)


# ## Compile kosmos-1408 list
# 
# first we devide the results into 4 categores:
# ```
# uncorrelated results
# | - possible kosmos fragments
# | - not kosmos fragments
# correlated results
# | - kosmos fragments
# | - not kosmos fragments
# ```
# 
# To do this we need selection criteria for "possible kosmos fragments" for each campagin

# In[4]:


class Categories:
    uncorrelated_other = 0
    correlated_other = 1
    uncorrelated_kosmos = 2
    correlated_kosmos = 3
category_names = [
    'Uncorrelated background',
    'Correlated background',
    'Uncorrelated KOSMOS-1408',
    'Correlated KOSMOS-1408',
]
# uncorrelated results: diamonds
# correlated results: crosses
# not kosmos: black
# kosmos: magenta
# box: possible kosmos selection region, red
styles = ['.k', 'xk', '.m', 'xm']
radar_title = {
    'esr': [
        'EISCAT Svalbard Radar 32m',
        'EISCAT Svalbard Radar 32m',
        'EISCAT Svalbard Radar 42m',
        'EISCAT Svalbard Radar 42m',
    ],
    'uhf': ['EISCAT UHF']*3,
}

# ### Category selection
# We can then select these categories, we start by loading the relevant data

# In[5]:


def calculate_categories(radar, date_ind, kosmos_selection):
    data = {}
    with h5py.File(paths['data_paths'][radar][date_ind], 'r') as hf:
        for key in hf:
            data[key] = hf[key][()]
    select = np.load(paths['correlation_select'][radar][date_ind])
    kosmos_select = np.load(paths['kosmos_correlation_select'][radar][date_ind])
    date = paths['data_paths'][radar][date_ind].stem.replace('.', '-')
    
    # The category identifier
    category = np.full((len(data['t'],)), Categories.uncorrelated_other, dtype=int)

    #get data
    r_box = kosmos_selection[radar]['r']

    t = (data['t'] - np.min(data['t']))/3600.0
    r = data['r']
    v = data['v']
    box_selectors = []
    t_boxes = []
    v_boxes = []
    for t_box, v_box in zip(
                kosmos_selection[radar]['t'][date_ind],
                kosmos_selection[radar]['v'][date_ind]
            ):
        t_boxes.append(t_box)
        v_boxes.append(v_box)
        box_selectors += [np.logical_and.reduce([
            t >= t_box[0], 
            t <= t_box[1],
            r >= r_box[0], 
            r <= r_box[1],
            v >= v_box[0], 
            v <= v_box[1],
        ])]
    box_selector = np.logical_or.reduce(box_selectors)

    # Assign categories
    category[:] = Categories.uncorrelated_other
    category[box_selector] = Categories.uncorrelated_kosmos
    category[select] = Categories.correlated_other
    category[kosmos_select] = Categories.correlated_kosmos
    
    out_categories = OUTPUT / f'{date}_{radar}_categories.npy'
    if not out_categories.is_file() or clobber:
        np.save(out_categories, category)
    
    return category, t_boxes, r_box, v_boxes, t, r, v


# In[9]:


def plot_cat(category, t_boxes, r_box, v_boxes, t, r, v, radar, date_ind):
    date = paths['data_paths'][radar][date_ind].stem.replace('.', '-')
    
    fig = plt.figure(figsize=(12,7))

    gs = GridSpec(2, 4, figure=fig)
    axes = [
        fig.add_subplot(gs[0, :3]),
        fig.add_subplot(gs[1, :3]),
    ]
    axes_z = [
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 3]),
    ]
    
    for ind, legend, style in zip(range(4), category_names, styles):
        axes[0].plot(t[category == ind], r[category == ind], style, label=legend)
        axes_z[0].plot(t[category == ind], r[category == ind], style, label=legend)
    axes[0].legend(ncol=2)
    for t_box, v_box in zip(t_boxes, v_boxes):
        rect = patches.Rectangle(
            (t_box[0], r_box[0]), 
            t_box[1] - t_box[0], 
            r_box[1] - r_box[0], 
            linewidth=1, 
            edgecolor='r', 
            facecolor='none',
        )
        axes[0].add_patch(rect)
    axes[0].set_ylabel('Range [km/s]')
    axes[0].set_ylim(None, 2500)
    axes[0].set_title(f'{radar_title[radar][date_ind]} beampark {date}')

    axes_z[0].set_xlim(*t_boxes[0])
    axes_z[0].set_ylim(*r_box)
    axes_z[0].set_title('Debris selection region')

    for ind, legend, style in zip(range(4), category_names, styles):
        axes[1].plot(t[category == ind], v[category == ind], style, label=legend)
        axes_z[1].plot(t[category == ind], v[category == ind], style, label=legend)
    for t_box, v_box in zip(t_boxes, v_boxes):
        rect = patches.Rectangle(
            (t_box[0], v_box[0]), 
            t_box[1] - t_box[0], 
            v_box[1] - v_box[0], 
            linewidth=1, 
            edgecolor='r', 
            facecolor='none',
        )
        axes[1].add_patch(rect)
    axes[1].set_ylim(-2.4, 2.4)
    axes[1].set_xlabel('Epoch + time [h]')
    axes[1].set_ylabel('Doppler velocity [km/s]')

    axes_z[1].set_xlim(*t_boxes[0])
    axes_z[1].set_ylim(*v_boxes[0])
    axes_z[1].set_xlabel('Epoch + time [h]')
    
    fig.savefig(out_plots / f'{date}_{radar}_rv_scatter.{fig_format}')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(4), [np.sum(category == ind) for ind in range(4)])
    ax.set_xticks(range(4), labels=[x.replace(' ', '\n') for x in category_names])
    ax.set_ylabel('Detections')
    ax.set_title(f'{radar_title[radar][date_ind]} beampark {date}')
    
    fig.savefig(out_plots / f'{date}_{radar}_categories.{fig_format}')


# In[10]:


#Structure is (lower left corner), (upper right corner)
# THESE WERE HANDPICKED
kosmos_selection = {
    'uhf': {
        'r': (280, 700),
        'v': [
            [(0.2, 1.6), ], 
            [(0.2, 1.6), ], 
            [(0.2, 1.6), (-0.4, 1), ], 
        ],
        't': [
            [(2.5, 3), ],
            [(2.1, 2.6), ],
            [(4.5, 5.2), (19.2, 19.8), ],
        ],
    },
    'esr': {
        'r': (280, 700),
        'v': [
            [(0.3, 1.6), (0.3, 1.6), ], 
            [(0.2, 1.6), ], 
            [(0.2, 1.6), ], 
            [(0.2, 1.6), ], 
        ],
        't': [
            [(9.7, 10.6), (16.6, 17.6)],
            [(2.4, 2.75), ],
            [(1.65, 1.8), ],
            [(2.75, 3.2), ],
        ],
    },
}


# In[11]:

for radar in radar_title:
    for ind in range(len(radar_title[radar])):
        category, t_boxes, r_box, v_boxes, t, r, v = calculate_categories(radar, ind, kosmos_selection)
        plot_cat(category, t_boxes, r_box, v_boxes, t, r, v, radar, ind)



