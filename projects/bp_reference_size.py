from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time, TimeDelta

import sorts
import pyant
from sorts.population import master_catalog, master_catalog_factor

epoch = Time('2021-11-23 12:00:00', format='iso')
master_path = Path.home() / 'data/master_2009/celn_20090501_00.sim'
t_stop = 3600.0
snr_lim = 10.0

# fig_format = 'png'
fig_format = 'eps'

HERE = Path(__file__).parent.resolve()
OUTPUT = HERE / 'output' / 'russian_asat'
print(f'Using {OUTPUT} as output')

out_pth = OUTPUT / 'size_reference_simulation'
out_pth.mkdir(exist_ok=True)
sim_pth = out_pth / 'master_sim'

pop_file = out_pth / 'master.h5'

options = dict(
    settings = dict(
        in_frame='TEME',
        out_frame='ITRS',
    ),
)

if pop_file.is_file():
    print(f'Using cached population: {pop_file}')
    pop = sorts.Population.load( 
        pop_file, 
        propagator = sorts.propagator.Kepler, 
        propagator_options = options,
    )
else:
    pop = master_catalog(master_path)
    pop = master_catalog_factor(pop, treshhold = 0.01)
    pop.filter('a', lambda x: x < 10e6)
    pop.filter('i', lambda x: x > 50.0)
    pop.save(pop_file)


class ForwardModel(sorts.Simulation):
    def __init__(self, population, epoch, *args, **kwargs):
        self.population = population
        self.epoch = epoch
        self.inds = list(range(len(population)))
        super().__init__(*args, **kwargs)

        self.cols = ['t', 'range', 'range_rate', 'snr']
        self.k_vars = ['kx', 'ky', 'kz']
        self._dtype = [('oid', np.int64)]
        self._dtype += [(key, np.float64) for key in self.cols]
        self._dtype += [(key, np.float64) for key in self.k_vars]
        self.collected_data = np.empty((0,), dtype=self._dtype)

        self.steps['simulate'] = self.simulate
        self.steps['collect'] = self.collect

    def extract_data(self, index, data):
        all_datas = []
        pass_data = data[0][0]
        if pass_data is not None and len(pass_data) > 0:
            for ps in pass_data:
                if ps is None:
                    continue
                all_datas.append(ps)

        new_data = np.empty((len(all_datas),), dtype=self._dtype)
        new_data['oid'] = index
        for ind, pd in enumerate(all_datas):

            max_snr_ind = np.argmax(pd['snr'])
            for key in self.cols:
                new_data[key][ind] = pd[key][max_snr_ind]
            for ki, key in enumerate(self.k_vars):
                new_data[key][ind] = pd['tx_k'][ki, max_snr_ind]
        return new_data

    def save_collected(self, **kwargs):
        np.save(self.get_path('collected_results.npy'), self.collected_data)

    @sorts.MPI_action(action='barrier')
    @sorts.iterable_step(iterable='inds', MPI=True, log=True, reduce=lambda x, y: None)
    @sorts.cached_step(caches='npy')
    def simulate(self, index, item, **kwargs):
        obj = self.population.get_object(item)
        t = np.arange(0.0, t_stop, 10.0)

        state = obj.get_state(t)
        t_rel = self.epoch - obj.epoch + TimeDelta(t, format='sec')
        interpolator = sorts.interpolation.Legendre8(state, t_rel.sec)

        passes = self.scheduler.radar.find_passes(
            t, 
            state, 
            cache_data = False,
        )

        data = self.scheduler.observe_passes(
            passes, 
            space_object = obj, 
            epoch = self.epoch, 
            interpolator = interpolator,
            snr_limit = False,
        )

        peak_data = self.extract_data(index, data)

        return peak_data

    @sorts.MPI_action(action='barrier')
    @sorts.MPI_single_process(process_id = 0)
    @sorts.pre_post_actions(post='save_collected')
    @sorts.iterable_cache(steps='simulate', caches='npy', MPI=False, log=True, reduce=lambda x, y: None)
    def collect(self, index, item, **kwargs):
        self.collected_data = np.append(self.collected_data, item)


class ObservedScanning(
                sorts.scheduler.StaticList,
                sorts.scheduler.ObservedParameters,
            ):
    pass


radar = sorts.radars.eiscat_uhf
radar.rx = radar.rx[:1]
tx = radar.tx[0]

scan = sorts.radar.scans.Beampark(
    azimuth=90.0, 
    elevation=75.0,
    dwell=0.1,
)

scanner = sorts.controller.Scanner(
    radar,
    scan,
    t = np.arange(0, t_stop, scan.dwell()),
    r = np.array([500e3]),
    t_slice = scan.dwell(),
)

scheduler = ObservedScanning(
    radar = radar, 
    controllers = [scanner], 
)

sim = ForwardModel(pop, epoch, scheduler, sim_pth)
sim_data_file = sim.get_path('collected_results.npy')

if not sim_data_file.is_file():
    sim.run()
    exit()

sim_data = np.load(sim_data_file)

keep = sim_data['snr'] > snr_lim
objs = sim_data['oid'][keep]

keep_big = np.logical_and(keep, pop['d'][sim_data['oid']] > 1e-1)
big_objs = sim_data['oid'][keep_big]

k_vecs = np.empty((3, len(sim_data['kx'])), dtype=np.float64)
k_vecs[0, :] = sim_data['kx']
k_vecs[1, :] = sim_data['ky']
k_vecs[2, :] = sim_data['kz']

tx.beam.sph_point(
    azimuth=90, 
    elevation=75,
)

fig, ax = plt.subplots(figsize=(12, 8))
pyant.plotting.gain_heatmap(tx.beam, min_elevation=85.0, ax=ax)
ax.plot(
    k_vecs[0, keep], 
    k_vecs[1, keep], 
    '.k', 
)
ax.set_title('Detected peak SNR location')
ax.set_xlabel('kx [1]')
ax.set_ylabel('ky [1]')
fig.savefig(out_pth / f'observed_k_vecs.{fig_format}')
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 8))
ax2 = ax.twinx()

zenith_ang = pyant.coordinates.vector_angle(tx.beam.pointing, k_vecs[:, keep], radians=False)

beam0 = tx.beam.copy()

beam0.sph_point(
    azimuth=0, 
    elevation=90,
)
theta = np.linspace(90, 85, num=1000)
_kv = np.zeros((3, len(theta)))
_kv[1, :] = theta
_kv[2, :] = 1
_kv = pyant.coordinates.sph_to_cart(_kv, radians=False)
S = beam0.gain(_kv)

p1, = ax2.plot(90 - theta, np.log10(S)*10.0, color='r')
ax.hist(zenith_ang)

ax2.yaxis.label.set_color(p1.get_color())
ax2.tick_params(axis='y', colors=p1.get_color())
ax.set_title('Detected peak SNR location')
ax.set_xlabel('Off-axis angle [deg]')
ax.set_ylabel('Detected objects')
ax2.set_ylabel('Gain [dB]')
fig.savefig(out_pth / f'detected_offaxis_angles.{fig_format}')
plt.close(fig)


fig, ax = plt.subplots(figsize=(12, 8))
ax2 = ax.twinx()

zenith_ang = pyant.coordinates.vector_angle(tx.beam.pointing, k_vecs[:, keep_big], radians=False)
p1, = ax2.plot(90 - theta, np.log10(S)*10.0, color='r')
ax.hist(zenith_ang)

ax2.yaxis.label.set_color(p1.get_color())
ax2.tick_params(axis='y', colors=p1.get_color())
ax.set_title('Detected peak SNR location')
ax.set_xlabel('Off-axis angle [deg]')
ax.set_ylabel('Detected objects')
ax2.set_ylabel('Gain [dB]')
fig.savefig(out_pth / f'detected_big_diam_offaxis_angles.{fig_format}')
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))
_, bins, _ = ax.hist(np.log10(pop['d'][objs]*1e2), 2*int(np.sqrt(len(objs))))
ax.set_xlabel('Diameter [log10(cm)]')
ax.set_ylabel('Detected objects')
fig.suptitle('MASTER-2008 EISCAT UHF simulation')
fig.savefig(out_pth / f'master_observed_size_dist.{fig_format}')
plt.close(fig)

fig, ax = plt.subplots()
ax.hist(np.log10(pop['d'][big_objs]*1e2), bins=bins)
ax.set_xlabel('Diameter [log10(cm)]')
ax.set_ylabel('Detected objects')
fig.savefig(out_pth / f'master_observed_big_diam_size_dist.{fig_format}')
plt.close(fig)

fig, ax = plt.subplots()
ax.scatter(np.log10(pop['d'][objs]*1e2), np.log10(sim_data['snr'][keep])*10, 2)
ax.set_xlabel('Diameter [log10(cm)]')
ax.set_ylabel('SNR [dB]')
fig.savefig(out_pth / f'master_observed_size_vs_snr.{fig_format}')
plt.close(fig)

fig, ax = plt.subplots()
ax.scatter(np.log10(pop['d'][sim_data['oid']]*1e2), np.log10(sim_data['snr'])*10, 2)
ax.axhline(np.log10(snr_lim)*10, color='r')
ax.set_xlabel('Diameter [log10(cm)]')
ax.set_ylabel('SNR [dB]')
fig.savefig(out_pth / f'master_size_vs_snr.{fig_format}')
plt.close(fig)

fig, ax = plt.subplots()
ax.hist(np.log10(pop['d']*1e2), 100)
ax.set_xlabel('Diameter [log10(cm)]')
ax.set_ylabel('Frequency')
fig.savefig(out_pth / f'master_size_dist.{fig_format}')
plt.close(fig)
