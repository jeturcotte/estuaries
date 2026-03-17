from __future__ import annotations

import argparse
from itertools import chain
from pathlib import Path

import pandas as pd


REPO_ROOT = Path( __file__ ).resolve( ).parents[ 1 ]
DEFAULT_INPUT = REPO_ROOT / 'Data/Team4D/team4d.baseline.1h.csv'
DEFAULT_OUTPUT = REPO_ROOT / 'Data/Team4D/team4d.baseline.4h.csv'
STATION_COL = 'meta_station_code'
TIME_COL = 'meta_datetime_stamp'
BIN_COL = '__bin_time'


def build_parser( ) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser( 
        description = 'Downsample the Team4D hourly baseline CSV to 4-hour resolution.',
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
        help = 'Path for the downsampled CSV.',
    )
    parser.add_argument( 
        '--freq',
        default = '4h',
        help = 'Pandas resample frequency string. Default: 4h.',
    )
    parser.add_argument( 
        '--chunksize',
        type = int,
        default = 200_000,
        help = 'Rows to process per chunk. Default: 200000.',
    )
    return parser


def load_data( source_path: Path, *, chunksize: int ) -> pd.io.parsers.TextFileReader:
    if not source_path.exists( ):
        raise FileNotFoundError( f'missing file: {source_path}' )

    return pd.read_csv( 
        source_path,
        parse_dates = [ TIME_COL ],
        low_memory = False,
        chunksize = chunksize,
    )


def split_columns( data: pd.DataFrame ) -> tuple[ set[ str ], list[ str ] ]:
    # Metadata and flags should be carried forward, not averaged.
    carry_cols = { 
        col
        for col in data.columns
        if col.startswith( 'meta_' ) and col not in { STATION_COL, TIME_COL }
    }
    carry_cols.update( col for col in data.columns if col.endswith( '_flag' ) )
    carry_cols.update( 
        { 
            'region_code',
            'm_nerrs_max_wspd_time_hhmm',
        }
    )

    value_cols = [ 
        col
        for col in data.columns
        if col not in carry_cols | { STATION_COL, TIME_COL }
    ]

    return carry_cols, value_cols


def aggregate_chunk( 
    chunk: pd.DataFrame,
    *,
    carry_cols: set[ str ],
    value_cols: list[ str ],
    freq: str,
) -> tuple[ pd.DataFrame, pd.DataFrame, pd.DataFrame ]:
    chunk = chunk.sort_values( [ STATION_COL, TIME_COL ] ).reset_index( drop = True )

    # Some measured fields contain strings like "<4>"; coerce them to NaN before aggregating.
    chunk[ value_cols ] = chunk[ value_cols ].apply( pd.to_numeric, errors = 'coerce' )
    chunk[ BIN_COL ] = chunk[ TIME_COL ].dt.floor( freq )

    grouped = chunk.groupby( 
        [ STATION_COL, BIN_COL ],
        dropna = False,
        sort = False,
    )

    # Average the measured values within each 4-hour window later via sum / count.
    value_sums = grouped[ value_cols ].sum( ).astype( float )
    value_counts = grouped[ value_cols ].count( ).astype( float )

    # Keep the first non-null metadata and quality flag values per window.
    carry_first = grouped[ sorted( carry_cols ) ].first( )

    return carry_first, value_sums, value_counts


def merge_group_rows( 
    left_carry: pd.DataFrame,
    left_sums: pd.DataFrame,
    left_counts: pd.DataFrame,
    right_carry: pd.DataFrame,
    right_sums: pd.DataFrame,
    right_counts: pd.DataFrame,
) -> tuple[ pd.DataFrame, pd.DataFrame, pd.DataFrame ]:
    merged_carry = left_carry.combine_first( right_carry )
    merged_sums = left_sums.add( right_sums, fill_value = 0 )
    merged_counts = left_counts.add( right_counts, fill_value = 0 )

    return merged_carry, merged_sums, merged_counts


def finalize_groups( 
    carry_data: pd.DataFrame,
    value_sums: pd.DataFrame,
    value_counts: pd.DataFrame,
    *,
    column_order: list[ str ],
) -> pd.DataFrame:
    value_means = value_sums.divide( value_counts.where( value_counts != 0 ) )

    finalized = ( 
        pd.concat( [ carry_data, value_means ], axis = 1 )
        .reset_index( )
        .rename( columns = { BIN_COL: TIME_COL } )
    )

    return finalized[ column_order ]


def write_groups( 
    carry_data: pd.DataFrame,
    value_sums: pd.DataFrame,
    value_counts: pd.DataFrame,
    *,
    column_order: list[ str ],
    output_path: Path,
    header: bool,
) -> int:
    if carry_data.empty:
        return 0

    finalized = finalize_groups( 
        carry_data,
        value_sums,
        value_counts,
        column_order = column_order,
    )
    finalized.to_csv( 
        output_path,
        mode = 'w' if header else 'a',
        header = header,
        index = False,
    )

    return len( finalized )


def main( ) -> None:
    args = build_parser( ).parse_args( )
    source_path = args.input.resolve( )
    target_path = args.output.resolve( )

    chunk_reader = load_data( source_path, chunksize = args.chunksize )

    try:
        first_chunk = next( chunk_reader )

    except StopIteration:
        raise ValueError( f'input file is empty: {source_path}' ) from None

    carry_cols, value_cols = split_columns( first_chunk )
    column_order = first_chunk.columns.tolist( )

    target_path.parent.mkdir( parents = True, exist_ok = True )
    if target_path.exists( ):
        target_path.unlink( )

    total_rows = 0
    total_output_rows = 0
    write_header = True
    pending_carry: pd.DataFrame | None = None
    pending_sums: pd.DataFrame | None = None
    pending_counts: pd.DataFrame | None = None

    for chunk in chain( [ first_chunk ], chunk_reader ):
        total_rows += len( chunk )
        carry_chunk, sums_chunk, counts_chunk = aggregate_chunk( 
            chunk,
            carry_cols = carry_cols,
            value_cols = value_cols,
            freq = args.freq,
        )

        if pending_carry is not None and not carry_chunk.empty:
            if pending_carry.index[ 0 ] == carry_chunk.index[ 0 ]:
                merged = merge_group_rows( 
                    pending_carry,
                    pending_sums,
                    pending_counts,
                    carry_chunk.iloc[ [ 0 ] ],
                    sums_chunk.iloc[ [ 0 ] ],
                    counts_chunk.iloc[ [ 0 ] ],
                )
                pending_carry, pending_sums, pending_counts = merged
                carry_chunk = carry_chunk.iloc[ 1: ]
                sums_chunk = sums_chunk.iloc[ 1: ]
                counts_chunk = counts_chunk.iloc[ 1: ]

            elif pending_carry.index[ 0 ] > carry_chunk.index[ 0 ]:
                raise ValueError( 'Input file is not ordered by station and timestamp.' )

        if pending_carry is not None and not carry_chunk.empty:
            total_output_rows += write_groups( 
                pending_carry,
                pending_sums,
                pending_counts,
                column_order = column_order,
                output_path = target_path,
                header = write_header,
            )
            write_header = False
            pending_carry = None
            pending_sums = None
            pending_counts = None

        elif pending_carry is not None and carry_chunk.empty:
            carry_chunk = pending_carry
            sums_chunk = pending_sums
            counts_chunk = pending_counts
            pending_carry = None
            pending_sums = None
            pending_counts = None

        if carry_chunk.empty:
            continue

        if len( carry_chunk ) > 1:
            total_output_rows += write_groups( 
                carry_chunk.iloc[ :-1 ],
                sums_chunk.iloc[ :-1 ],
                counts_chunk.iloc[ :-1 ],
                column_order = column_order,
                output_path = target_path,
                header = write_header,
            )
            write_header = False

        pending_carry = carry_chunk.iloc[ [ -1 ] ]
        pending_sums = sums_chunk.iloc[ [ -1 ] ]
        pending_counts = counts_chunk.iloc[ [ -1 ] ]

    if pending_carry is not None:
        total_output_rows += write_groups( 
            pending_carry,
            pending_sums,
            pending_counts,
            column_order = column_order,
            output_path = target_path,
            header = write_header,
        )

    row_drop_pct = 100 * ( 1 - total_output_rows / total_rows )
    original_size_mb = source_path.stat( ).st_size / 1024**2
    downsampled_size_mb = target_path.stat( ).st_size / 1024**2

    print( f'saved: {target_path}' )
    print( f'rows: {total_rows:,} -> {total_output_rows:,} ({row_drop_pct:.1f}% fewer)' )
    print( f'size: {original_size_mb:.1f} MB -> {downsampled_size_mb:.1f} MB' )


if __name__ == '__main__':
    main( )
