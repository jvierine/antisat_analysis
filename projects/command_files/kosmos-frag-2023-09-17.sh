# Convert regular leo results to h5
python convert_spade_events_to_h5.py ~/data/EISCAT/leo_bpark_2023_09_17/{LEO_RESULTS_20230917,leo.h5}

# download spacetrack catalog
sorts spacetrack -k space-track 24h 2023-09-17 ~/data/EISCAT/leo_bpark_2023_09_17/space-track.tles

# Run correlation
# mpirun -n 4 ./beampark_correlator.py \
#     -c --jitter --target-epoch "2023-09-17 14:00:00" eiscat_uhf \
#     ~/data/EISCAT/leo_bpark_2023_09_17/{space-track.tles,leo.h5,correlation.h5}
mpirun -n 4 ./beampark_correlator.py \
    eiscat_uhf \
    ~/data/EISCAT/leo_bpark_2023_09_17/{space-track.tles,leo.h5,correlation.h5}

mkdir ~/data/EISCAT/leo_bpark_2023_09_17/plots

# Plot correlation results
python beampark_correlation_analysis.py \
    --output ~/data/EISCAT/leo_bpark_2023_09_17/plots \
    ~/data/EISCAT/leo_bpark_2023_09_17/{correlation.h5,leo.h5}


# download spacetrack catalog with just starlink
sorts spacetrack -k space-track -n starlink 24h 2023-09-17 ~/data/EISCAT/leo_bpark_2023_09_17/space-track-starlink.tles

# Pick out star-link
python projects/object_select.py \
    --output ~/data/EISCAT/leo_bpark_2023_09_17/plots \
    --name "STARLINK" \
    ~/data/EISCAT/leo_bpark_2023_09_17/{\
        leo.h5,\
        correlation.h5,\
        space-track.tles,\
        space-track-starlink.tles,\
        plots/eiscat_uhf_selected_correlations.npy}