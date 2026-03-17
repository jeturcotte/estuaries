from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path( __file__ ).resolve( ).parents[ 1 ]
DEFAULT_INPUT = REPO_ROOT / 'data/1hr/t4d.1hr.baseline.csv'
DEFAULT_OUTPUT = REPO_ROOT / 'data/12hr/t4d.12hr.baseline.csv'
TIME_COL = 'meta_datetime_stamp'
KEEP_HOURS = { 8, 20 }


def build_parser( ) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser( 
        description = 'Keep only 8am and 8pm rows from the Team4D hourly baseline CSV.',
    )
    parser.add_argument( 
        '--input',
        type = Path,
        default = DEFAULT_INPUT,
        help = 'Path to the hourly baseline CSV.',
    )
    parser.add_argument( 
        '--output',
        type = Path,
        default = DEFAULT_OUTPUT,
        help = 'Path for the filtered CSV.',
    )
    parser.add_argument( 
        '--chunksize',
        type = int,
        default = 200_000,
        help = 'Rows to process per chunk. Default: 200000.',
    )
    return parser


def stream_filtered_rows( 
    source_path: Path,
    *,
    chunksize: int,
) -> tuple[ pd.io.parsers.TextFileReader, list[ str ] ]:
    if not source_path.exists( ):
        raise FileNotFoundError( f'missing file: {source_path}' )

    preview = pd.read_csv( source_path, nrows = 0 )
    reader = pd.read_csv( 
        source_path,
        parse_dates = [ TIME_COL ],
        low_memory = False,
        chunksize = chunksize,
    )

    return reader, preview.columns.tolist( )


def main( ) -> None:
    args = build_parser( ).parse_args( )
    source_path = args.input.resolve( )
    target_path = args.output.resolve( )

    reader, columns = stream_filtered_rows( 
        source_path,
        chunksize = args.chunksize,
    )

    target_path.parent.mkdir( parents = True, exist_ok = True )
    if target_path.exists( ):
        target_path.unlink( )

    total_rows = 0
    kept_rows = 0
    wrote_header = False

    for chunk in reader:
        total_rows += len( chunk )

        keep_mask = chunk[ TIME_COL ].dt.hour.isin( KEEP_HOURS )
        filtered = chunk.loc[ keep_mask, columns ]

        if filtered.empty:
            continue

        filtered.to_csv( 
            target_path,
            mode = 'a',
            header = not wrote_header,
            index = False,
        )
        wrote_header = True
        kept_rows += len( filtered )

    if not wrote_header:
        pd.DataFrame( columns = columns ).to_csv( target_path, index = False )

    kept_pct = 100 * kept_rows / total_rows if total_rows else 0.0
    original_size_mb = source_path.stat( ).st_size / 1024**2
    filtered_size_mb = target_path.stat( ).st_size / 1024**2

    print( f'saved: {target_path}' )
    print( f'rows kept: {kept_rows:,} of {total_rows:,} ({kept_pct:.1f}%)' )
    print( f'size: {original_size_mb:.1f} MB -> {filtered_size_mb:.1f} MB' )


if __name__ == '__main__':
    main( )
