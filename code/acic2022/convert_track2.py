import pathlib
import re

import polars as pl
import tqdm

""" makes practice_year_aux/ from the track1 ACIC 2022 data challenge data,
takes about 1 hour reading from magnetic disk """

# config
sourceroot = pathlib.Path('...') / 'ACIC_2022_track1'
destdir = pathlib.Path('acic2022') / 'practice_year_aux'
columns = 'V1', 'V2', 'V4'

# check directories
assert sourceroot.is_dir()
destdir.mkdir(exist_ok=True)

for subdir in sourceroot.iterdir():

    # check subdirectory is valid
    if not subdir.is_dir() and not re.fullmatch(r'track1\w_\d{8}', subdir.name):
        continue
    print(f'Processing {subdir.name}...')

    # iterate over datasets
    for file in tqdm.tqdm(list((subdir / 'patient').iterdir())):

        # conspecific files
        m = re.fullmatch(r'acic_patient_(\d{4})\.csv', file.name)
        dataset = int(m.group(1))
        year_file = file.parent.with_name('patient_year') / f'acic_patient_year_{dataset:04d}.csv'
        out_file = destdir / f'acic_practice_year_{dataset:04d}.arrow'
        if out_file.exists():
            continue

        # load data
        df_patient = pl.read_csv(file)
        df_year = pl.read_csv(year_file)
        df = df_patient.join(df_year, on='id.patient')

        # compute summaries
        df = (df
            .group_by('id.practice', 'year')
            .agg(
                pl.col(columns).std().name.suffix('_std'),
                pl.col(columns).min().name.suffix('_min'),
                pl.col(columns).max().name.suffix('_max'),
            )
            .sort('id.practice', 'year')
        )

        # sanity checks
        assert len(df) == 500 * 4
        assert 'id.practice' in df.columns
        assert 'year' in df.columns
        assert df.width == 2 + len(columns) * 3

        # save
        df.write_ipc(out_file, compression='zstd')
