import os
import shutil
path = '/projects/illinois/ovcri/beckman/aandrsn3/'

for folder in next(os.walk(path))[1]:

    print(path)
    sigFiles = [f for f in os.listdir(path + folder) if f.endswith('_EEG.mat')]
    os.makedirs(f"{path}{folder}/part_1", exist_ok=True)
    os.makedirs(f"{path}{folder}/part_2", exist_ok=True)
    os.makedirs(f"{path}{folder}/part_3", exist_ok=True)
    os.makedirs(f"{path}{folder}/part_4", exist_ok=True)

    tot = len(sigFiles)
    if tot < 4:
        continue
    quart = (tot - 1)//4 + 1

    for i, f in enumerate(sigFiles):
        shutil.copy(f'{path}{folder}/{f}', f'{path}{folder}/part_{i//quart+1}/{f}')
        g = f.replace(".mat", ".hea")
        shutil.copy(f'{path}{folder}/{g}', f'{path}{folder}/part_{i//quart+1}/{g}')