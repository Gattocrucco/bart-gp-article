import pathlib
import datetime
import subprocess
import contextlib
import time
import re
import sys

import numpy as np
import tqdm

""" runs the synthetic data test, takes about 1 day """

# config
root = pathlib.Path('acic2022/loop')
task = root / 'task.py'
other_scripts = [root / 'analysis.py']
workparent = root / 'workdirs'
tablefilename = 'results.npy'
logfilename = 'log.txt'
workdirprefix = 'workdir_'
completioncanary = 'DONE'
ndatasets = 850 # between 1 and 3400
baseseed = 202311102012
timeout = 170
maxretries = 3

# list working dirs
workparent.mkdir(exist_ok=True)
workdirs = sorted(
    p for p in workparent.iterdir()
    if p.is_dir() and re.fullmatch(fr'{workdirprefix}\d{{14}}', p.name)
)

# fetch last one and check if it contains completed work    
if not workdirs or (workdirs[-1] / completioncanary).exists():
    # create new workdir with current date
    workdir = workparent / f'{workdirprefix}{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
    assert not workdir.exists()
    workdir.mkdir()
else:
    workdir = workdirs[-1]

# create table of results if it does not exist
tablefile = workdir / tablefilename
if not tablefile.exists():
    table = np.zeros(ndatasets, [
        ('dataset', int),
        ('done', bool),
        ('seed', int, 2),
        ('time_total', float),
        ('time_ps', float),
        ('time_hypers', float),
        ('time_satt', float),
        ('retries', int),
        ('utc_timestamp', int),
    ] + sum([
        [
            (f'{level}_satt', float),
            (f'{level}_lower90', float),
            (f'{level}_upper90', float),
            (f'{level}_true', float),
        ]
        for level in [
            'Overall',
            'Yearly=3',
            'Yearly=4',
            'X1=0',
            'X1=1',
            'X2=A',
            'X2=B',
            'X2=C',
            'X3=0',
            'X3=1',
            'X4=A',
            'X4=B',
            'X4=C',
            'X5=0',
            'X5=1',
        ]
    ], start=[]))
    table['dataset'] = 1 + np.arange(ndatasets)
    np.save(tablefile, table)

# use pip to write a requirements file
with open(workdir / 'requirements.txt', 'w') as f:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f)

# copy the scripts into the directory
for script in [pathlib.Path(__file__), task, *other_scripts]:
    scriptcopy = workdir / script.name
    scriptcopy.write_text(script.read_text())

class Timer:
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *_):
        self.duration = time.perf_counter() - self.start

# mark log file
with open(workdir / logfilename, 'a') as logfile:
    logfile.write(f'\n\n\n\n################### NEW RUN ####################\n')
    logfile.write(f'{datetime.datetime.now()}\n')

# load table as memmap to write results progressively
table = np.load(tablefile, mmap_mode='r+')
try: # try block to close memmap in any case, can't use context manager

    # fetch entries still to do and randomize order to balance ETA
    indices = np.flatnonzero(~table['done'])
    np.random.default_rng(baseseed).shuffle(indices)

    skipped = []
    
    # cycle over todo entries
    for idx in tqdm.tqdm(indices):
        entry = table[idx]

        for retries in range(maxretries):
            
            with (open(workdir / logfilename, 'a', buffering=1) as logfile,
                contextlib.redirect_stdout(logfile)):

                print(f'\n\n@@@@@@@@@@@@@@@@@@@@@@@@ DATASET {entry["dataset"]} @@@@@@@@@@@@@@@@@@@@@@@@')
                print(f'{datetime.datetime.now()}\n')
                timestamp = datetime.datetime.now(datetime.UTC).timestamp()
                
                try:
                    with Timer() as timer_total:
                        subprocess.run([
                            sys.executable,
                            task,
                            '-t', tablefile,
                            '-i', str(idx),
                            '-s', str(baseseed),
                        ], timeout=timeout, check=True, stdout=sys.stdout)
                
                except subprocess.TimeoutExpired:
                    print('\n###### ! TIMEOUT ! #######\n')

                except subprocess.CalledProcessError:
                    print('\n###### ! ERROR ! #######\n')

                else:
                    entry['time_total'] = timer_total.duration
                    entry['retries'] = retries
                    entry['utc_timestamp'] = timestamp
                    entry['done'] = True
                    break

        else:
            print(f'###### ! skipping entry {idx} ! ######')
            skipped.append(idx)

        # save to disk
        table.flush()

    # mark work as completed
    if skipped:
        print(f'\nSkipped entries: {skipped}')
    else:
        (workdir / completioncanary).touch()

finally:
    del table # to close writable memmap
