from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time, TimeDelta

import sorts
from sorts.population import master_catalog, master_catalog_factor

epoch = Time('2021-11-23 12:00:00', format='iso')
master_path = Path.home() / 'data/master_2009/celn_20090501_00.sim'
t_stop = 3600.0
snr_limit = 33.0

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
        self._dtype = [('oid', np.int64)] + [(key, np.float64) for key in self.cols]
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

keep = sim_data['snr'] > snr_limit
objs = sim_data['oid'][keep]

fig, ax = plt.subplots()
ax.hist(np.log10(pop['d'][objs]*1e2), 100)
ax.set_xlabel('Diameter [log10(cm)]')
ax.set_ylabel('Frequency')
fig.savefig(out_pth / 'master_observed_size_dist.png')
plt.close(fig)


fig, ax = plt.subplots()
ax.hist(np.log10(pop['d']*1e2), 100)
ax.set_xlabel('Diameter [log10(cm)]')
ax.set_ylabel('Frequency')
fig.savefig(out_pth / 'master_size_dist.png')
plt.close(fig)
