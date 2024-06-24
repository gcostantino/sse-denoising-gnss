import os
import re

import numpy as np
import requests

cascadia_box = [40, 51.8, -128.3, -121]  # min/max_latitude, min/max_longitude

work_directory = os.path.expandvars('$WORK')
work_directory = './geo_data'

geo_path = './geo_data'
base_path = work_directory  # './geo_data'
gnss_path = os.path.join(base_path, 'GNSS_CASCADIA')
# tenv_path = os.path.join(gnss_path, 'tenv')
tenv3_path = os.path.join(gnss_path, 'tenv3')
txt_path = os.path.join(gnss_path, 'txt')

if not os.path.exists(gnss_path):
    os.makedirs(gnss_path)

if not os.path.exists(tenv3_path):
    os.makedirs(tenv3_path)

if not os.path.exists(txt_path):
    os.makedirs(txt_path)

# we download the time series for the chosen cascadia gnss stations
with open(os.path.join(geo_path, 'NGL_stations_cascadia.txt')) as f:
    data_all = f.read().splitlines()

codes = []
for i, line in enumerate(data_all):
    codes.append(line.split(' ')[0])

for i, code in enumerate(codes):
    if i % 50 == 0:
        print(f"{int(i / len(codes) * 100)}% completed")
    # response = requests.get(f"http://geodesy.unr.edu/gps_timeseries/tenv/IGS14/{code}.tenv")
    response = requests.get(f"http://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/{code}.tenv3")
    with open(os.path.join(tenv3_path, f'{code}.txt'), 'wb') as f:
        f.write(response.content)

# we convert the tenv files in txt with the displacement values only -> columns: decimal year, dE, dN, dU


for i, code in enumerate(codes):
    if i % 50 == 0:
        print(f"{int(i / len(codes) * 100)}% completed")
    with open(os.path.join(tenv3_path, f'{code}.txt'), 'r') as fread:
        with open(os.path.join(txt_path, f'{code}.txt'), 'w') as fwrite:
            data_all = fread.read().splitlines()[1:]  # skip header
            ref_line = re.sub(' +', ' ', data_all[0])
            # ref_data_line = np.array(ref_line.split(' '))[[2, 8, 10, 12]].astype(np.float_)
            ref_data_line = np.array(ref_line.split(' '))[[2, 7, 8, 9, 10, 11, 12]].astype(np.float_)
            ref_data_disp = ref_data_line[[0, 2, 4, 6]]
            ref_data_disp[0] = 0.
            decimal_part_ref = ref_data_line[[1, 3, 5]]

            for j in range(len(data_all)):
                line = re.sub(' +', ' ', data_all[j])
                data_line = np.array(line.split(' '))[[2, 7, 8, 9, 10, 11, 12]].astype(np.float_)
                disp_data_line = data_line[[0, 2, 4, 6]]
                decimal_part_line = data_line[[1, 3, 5]]
                decimal_diff = decimal_part_ref - decimal_part_line
                if decimal_diff.any() != 0.:
                    disp_data_line[1:] = disp_data_line[1:] - decimal_diff
                    # decimal_part_ref = decimal_part_line
                # data_line = np.array(line.split(' '))[[2, 6, 7, 8]]
                disp_data_line = disp_data_line - ref_data_disp
                fwrite.write(f"{disp_data_line[0]} {disp_data_line[1]} {disp_data_line[2]} {disp_data_line[3]}\n")
