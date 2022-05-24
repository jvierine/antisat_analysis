#!/usr/bin/env python
# coding: utf-8

# # This file details the step by step analysis of the russian asat
# 
# TODO: can this be a snake-make file? that would be a reproducible workflow

# In[1]:


from pathlib import Path
HERE = Path(__file__).parent.resolve()
OUTPUT = HERE / 'output' / 'russian_asat'
OUTPUT.mkdir(exist_ok=True)
print(f'Using {OUTPUT} as output')


# In[2]:


spade_data_paths = {
    'uhf': [
        '/home/danielk/data/spade/beamparks/uhf/2021.11.23',
        '/home/danielk/data/spade/beamparks/uhf/2021.11.25',
        '/home/danielk/data/spade/beamparks/uhf/2021.11.29',
    ],
    'esr': [
        '/home/danielk/data/spade/beamparks/esr/2021.11.19',
        '/home/danielk/data/spade/beamparks/esr/2021.11.23',
        '/home/danielk/data/spade/beamparks/esr/2021.11.25',
        '/home/danielk/data/spade/beamparks/esr/2021.11.29',
    ]
}

data_paths = {}
for key in spade_data_paths:
    data_paths[key] = []
    for pth in spade_data_paths[key]:
        data_paths[key].append(OUTPUT / key / (Path(pth).name + '.h5'))


# ## Convert event.txt to h5

# In[3]:


cmds = []
for key in spade_data_paths:
    for pth, out in zip(spade_data_paths[key], data_paths[key]):

        pth = Path(pth).resolve()
        if not out.parent.is_dir():
            out.parent.mkdir(parents=True, exist_ok=True)
        cmd = f'python convert_spade_events_to_h5.py {str(pth)} {str(out)}'
        cmds.append(cmd)
        if not out.is_file():
            print(cmd)
        else:
            print(f'{out} exsists, skipping...')


# In[4]:


print(' && '.join(cmds))


# In[5]:


from astropy.time import Time, TimeDelta
import h5py
import numpy as np

# Determine dates of files
dates = []
dates_keyed = {}
for key in data_paths:
    dates_keyed[key] = []
    for pth in data_paths[key]:
        with h5py.File(pth, 'r') as hf:
            epoch = Time(np.min(hf['t']), format='unix', scale='utc')
            dates.append(epoch.datetime.date())
            dates_keyed[key].append(epoch.datetime.date())

dates = np.unique(dates).tolist()


# In[6]:


for dt in dates:
    print(dt.strftime('%Y-%m-%d'))


# ## Download spacetrack catalog
# 
# Run the commands generate below to get the appropriate catalogs

# In[7]:


# Set this to None if you dont want to use a secret-tool to get credentials
#  they need to be stored under the "password" and "username" attributes
secret_tool_key = 'space-track'

catalog_commands = []
tle_catalogs = []
for dt in dates:
    dt_str = dt.strftime('%Y-%m-%d')
    
    out_tle = OUTPUT / f'{dt_str.replace("-","_")}_spacetrack.tle'
    tle_catalogs.append(out_tle)
    
    cmd = f'./space_track_download.py 24h {dt_str} {str(out_tle)}'
    if secret_tool_key is not None:
        cmd += f' -k {secret_tool_key}'
    
    print(cmd)
    catalog_commands.append(cmd)


# In[8]:


#Verify that catalogs exist
for pth in tle_catalogs:
    assert pth.is_file(), f'{pth} does not exist'
    print(f'{pth} exists...')


# ## Correlate measurnments
# 
# ### Calculate correlations
# 
# Choose number of cores to run on

# In[9]:


cores = 6
override = False
jitter = True
r_scaling = 1.0
dr_scaling = 0.2
save_states = False


# In[10]:


correlate_commands = []

override_cm = '-c' if override else ''
jitter_cm = '--jitter' if jitter else '' --save-states
save_states_cm = '--save-states' if save_states else '' 
r_scaling_cm = f'--range-scaling {r_scaling}'
dr_scaling_cm = f'--range-rate-scaling {dr_scaling}'

correlations = {}
for key in data_paths:
    correlations[key] = []
    for pth, dt in zip(data_paths[key], dates_keyed[key]):
        tle_file = tle_catalogs[dates.index(dt)]
        out_file = OUTPUT / (pth.stem + f'_{key}_correlation.pickle')
        
        correlations[key].append(out_file)
        
        cmd = f'mpirun -n {cores} ./beampark_correlator.py eiscat_{key} '
        cmd += f'{str(tle_file)} {str(pth)} {str(out_file)} {override_cm} '
        cmd += f'{jitter_cm} {r_scaling_cm} {dr_scaling_cm} {save_states_cm} -c'
        print(cmd + '\n')
        correlate_commands.append(cmd)


# In[11]:


#Verify that catalogs exist
for key in correlations:
    for pth in correlations[key]:
        assert pth.is_file(), f'{pth} does not exist'
        print(f'{pth} exists...')


# ### Analyse correalations
# 
# Set `threshold` to `None` to auto determine threshold. The unit of threshold is determined by the scaling that is applied to range in meter and range rate in meters per second.

# In[12]:


threshold = None #1.0e3


# In[13]:


correlation_analysis_cmds = []

threshold_cm = '' if threshold is None else f'--threshold {threshold}'

correlation_select = {}
for key in data_paths:
    correlation_select[key] = []
    for pth, cor in zip(data_paths[key], correlations[key]):
        out_folder = OUTPUT / (pth.stem + f'_{key}_correlation_plots')
        
        correlation_select[key].append(out_folder)
        
        cmd = f'./beampark_correlation_analysis.py --output {out_folder} '
        cmd += f'{threshold_cm} {r_scaling_cm} {dr_scaling_cm} '
        cmd += f'{cor} {pth}'
        
        print(cmd + '\n')
        correlation_analysis_cmds.append(cmd)


# In[14]:


print(' && '.join(correlation_analysis_cmds))


# In[15]:


for key in correlation_select:
    paths = []
    for pth in correlation_select[key]:
        if not pth.is_file():
            paths.append(list(pth.glob('*.npy'))[0])
        else:
            paths.append(pth)
    correlation_select[key] = paths


# ## Plot measurements & debris cloud
# 
# Hand-craft catalogs for kosmos debris

# In[16]:


meas_plot_cmds = []
for key in data_paths:
    for pth in data_paths[key]:
        out_folder = OUTPUT / (pth.stem + f'_{key}_measurement_plots')
        out_folder.mkdir(exist_ok=True)
        
        cmd = f'./beampark_plot.py -o {out_folder} -r "{key}" {pth}'
        meas_plot_cmds.append(cmd)
        
print(' && '.join(meas_plot_cmds))


# ### Download kosmos TLEs

# In[17]:


import datetime

dt_max = datetime.timedelta(days=2)
back = f'{(dt_max.days+1)*24}h'

kosmos_catalogs = []
kosmos_download_cmds = []
for dt in dates:
    outf = OUTPUT / f'{str(dt).replace("-", "_")}_spacetrack_kosmos.tle'
    kosmos_catalogs.append(outf)
    cmd = f'./space_track_download.py {back} {str(dt + dt_max)} '
    cmd += f'{outf} -k space-track -n "COSMOS 1408 DEB"'
    print('\n' + cmd)
    kosmos_download_cmds.append(cmd)


# In[19]:


print(' && '.join(kosmos_download_cmds))


# If not enough TLEs use the below to use one big file for all correlations (can be unstable)

# In[20]:


dt_max = datetime.timedelta(days=2)
back = f'{(dt_max.days+2)*24}h'
dt = dates[-1]
outf = OUTPUT / f'{str(dt).replace("-", "_")}_spacetrack_kosmos.tle'
cmd = f'./space_track_download.py {back} {str(dt + dt_max)} '
cmd += f'{outf} -k space-track -n "COSMOS 1408 DEB"'
print('\n' + cmd)
for fname in kosmos_catalogs[:-1]:
    print(f'cp -v {kosmos_catalogs[-1]} {fname}')


# In[21]:


for pth in kosmos_catalogs:
    assert pth.is_file(), f'{pth} does not exist'
    print(f'{pth} exists...')


# ### Correlate with kosmos debs

# In[22]:


kosmos_threshold = 3e3
kosmos_threshold_cm = '' if kosmos_threshold is None else f'--threshold {kosmos_threshold}'


# In[23]:


kosmos_correlate_commands = []
kosmos_correlations = {}
for key in data_paths:
    kosmos_correlations[key] = []
    for pth, dt in zip(data_paths[key], dates_keyed[key]):
        tle_file = kosmos_catalogs[dates.index(dt)]
        if tle_file.stat().st_size == 0:
            kosmos_correlations[key].append(None)
            print('no kosmos elements, skipping...')
            continue
        out_file = OUTPUT / (pth.stem + f'_{key}_kosmos_correlation.pickle')
        
        kosmos_correlations[key].append(out_file)
        
        cmd = f'./beampark_correlator.py eiscat_{key} '
        cmd += f'{str(tle_file)} {str(pth)} {str(out_file)} {override_cm} '
        cmd += f'--target-epoch {dt} '
        cmd += f'{jitter_cm} {r_scaling_cm} {dr_scaling_cm} {save_states_cm} -c'
        kosmos_correlate_commands.append(cmd)

print(' && '.join(kosmos_correlate_commands))
print('\n')  



kosmos_correlation_analysis_cmds = []
kosmos_correlation_select = {}
for key in data_paths:
    kosmos_correlation_select[key] = []
    for pth, cor in zip(data_paths[key], kosmos_correlations[key]):
        if cor is None:
            print('no kosmos correlation, skipping...')
            kosmos_correlation_select[key].append(None)
            continue
        out_folder = OUTPUT / (pth.stem + f'_{key}_kosmos_correlation_plots')
        
        kosmos_correlation_select[key].append(out_folder)
        
        cmd = f'./beampark_correlation_analysis.py --output {out_folder} '
        cmd += f'{kosmos_threshold_cm} {r_scaling_cm} {dr_scaling_cm} '
        cmd += f'{cor} {pth}'
        
        kosmos_correlation_analysis_cmds.append(cmd)
print(' && '.join(kosmos_correlation_analysis_cmds))
print('\n')  
# exit()
# In[25]:


for key in kosmos_correlation_select:
    paths = []
    for pth in kosmos_correlation_select[key]:
        if pth is None:
            paths.append(None)
            continue
        if not pth.is_file():
            paths.append(list(pth.glob('*.npy'))[0])
        else:
            paths.append(pth)
    kosmos_correlation_select[key] = paths


# ## Save paths
# now we save all paths in a convenient pickle file

# In[30]:


paths = {
    'data_paths': data_paths,
    'tle_catalogs': tle_catalogs,
    'correlations': correlations,
    'correlation_select': correlation_select,
    'kosmos_catalogs': kosmos_catalogs,
    'kosmos_correlations': kosmos_correlations,
    'kosmos_correlation_select': kosmos_correlation_select,
    'dates': dates,
}


# In[34]:


import pickle

with open(OUTPUT / 'paths.pickle', 'wb') as fh:
    pickle.dump(paths, fh)

