#!/usr/bin/env python3
"""bare bones collation for wq and nutrient files only"""

from __future__ import annotations

import argparse
import csv
import gc
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path( __file__ ).resolve( ).parents[ 2 ]
DEFAULT_DATA_DIR = PROJECT_ROOT / 'Data' / 'NERRS'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'Data' / 'NERRS'

WQ_OUTPUT_NAME = 'nerrs.wq.all.raw.csv'
NUT_OUTPUT_NAME = 'nerrs.nut.all.raw.csv'
MERGED_OUTPUT_NAME = 'nerrs.wq_nut.asof_7d.csv'
DEFAULT_JOIN_CHUNK_SIZE = 250000
NUTRIENT_VALUE_COLUMNS = [ 'PO4F', 'NH4F', 'NO2F', 'NO3F', 'NO23F', 'CHLA_N' ]

FILENAME_PATTERN = re.compile( 
    r'^(?P<station>[a-z0-9]{5})(?P<file_type>wq|nut)(?P<suffix>[a-z0-9_-]*)$',
    re.IGNORECASE,
)

DATETIME_FORMATS = ( 
    '%m/%d/%Y %H:%M',
    '%m/%d/%Y %H:%M:%S',
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d %H:%M',
)



@dataclass( frozen = True )



class SourceFile:
    path: Path
    region: str
    station: str
    file_type: str


def parse_args( ) -> argparse.Namespace:
    parser = argparse.ArgumentParser( description = 'collate all wq and nutrient rows into two csv files' )
    parser.add_argument( '--data-dir', type = Path, default = DEFAULT_DATA_DIR )
    parser.add_argument( '--output-dir', type = Path, default = DEFAULT_OUTPUT_DIR )
    parser.add_argument( '--join-chunk-size', type = int, default = DEFAULT_JOIN_CHUNK_SIZE )
    return parser.parse_args( )


def classify_filename( stem: str ) -> SourceFile | None:
    match = FILENAME_PATTERN.match( stem.lower( ) )
    if not match:
        return None

    station = match.group( 'station' )
    return SourceFile( 
        path = Path( '' ),
        region = station[ :3 ],
        station = station,
        file_type = match.group( 'file_type' ),
    )


def discover_source_files( data_dir: Path ) -> list[ SourceFile ]:
    # scan all csv files and keep only wq or nut naming
    discovered: list[ SourceFile ] = [ ]

    csv_paths = sorted( 
        [ 
            file_path
            for file_path in data_dir.rglob( '*' )
            if file_path.is_file( ) and file_path.suffix.lower( ) == '.csv'
        ],
    )

    for csv_path in csv_paths:
        parsed = classify_filename( csv_path.stem )
        if parsed is None:
            continue

        discovered.append( 
            SourceFile( 
                path = csv_path,
                region = parsed.region,
                station = parsed.station,
                file_type = parsed.file_type,
            ),
        )

    return discovered


def collect_fieldnames( files: Iterable[ SourceFile ] ) -> list[ str ]:
    # collect union of source headers so writer has one stable schema
    seen: set[ str ] = set( )
    ordered: list[ str ] = [ ]

    for source_file in files:
        with open( source_file.path, 'r', encoding = 'utf-8', errors = 'ignore' ) as handle:
            reader = csv.DictReader( handle )
            headers = reader.fieldnames or [ ]

        for header in headers:
            if not header:
                continue

            clean = header.strip( )
            if not clean or clean in seen:
                continue

            seen.add( clean )
            ordered.append( clean )

    return ordered


def is_hourly_timestamp( raw_value: str ) -> bool:
    # keep only full hour marks for wq rows
    if raw_value is None:
        return False

    value = raw_value.strip( )
    if not value:
        return False

    for fmt in DATETIME_FORMATS:
        try:
            parsed = datetime.strptime( value, fmt )
            return parsed.minute == 0 and parsed.second == 0

        except ValueError:
            continue

    return False


def write_collated_csv( files: list[ SourceFile ], output_path: Path ) -> tuple[ int, int ]:
    # write all rows no qaqc filtering and no transforms
    output_path.parent.mkdir( parents = True, exist_ok = True )

    source_fields = collect_fieldnames( files )
    out_fields = [ 'region', 'station', 'datetime', 'source_file' ] + source_fields

    row_count = 0
    skipped_non_hourly = 0

    with open( output_path, 'w', newline = '', encoding = 'utf-8' ) as out_handle:
        writer = csv.DictWriter( out_handle, fieldnames = out_fields, extrasaction = 'ignore' )
        writer.writeheader( )

        for source_file in files:
            with open( source_file.path, 'r', encoding = 'utf-8', errors = 'ignore' ) as in_handle:
                reader = csv.DictReader( in_handle )

                for row in reader:
                    raw_datetime = row.get( 'DateTimeStamp' ) or row.get( 'DatetimeStamp' ) or ''
                    if source_file.file_type == 'wq' and not is_hourly_timestamp( raw_datetime ):
                        skipped_non_hourly += 1
                        continue

                    clean_row = { field: row.get( field, '' ) for field in source_fields }
                    clean_row[ 'region' ] = source_file.region
                    clean_row[ 'station' ] = source_file.station
                    clean_row[ 'datetime' ] = raw_datetime
                    clean_row[ 'source_file' ] = source_file.path.name

                    writer.writerow( clean_row )
                    row_count += 1

            if row_count and row_count % 500000 == 0:
                print( f'- wrote {row_count} rows to {output_path.name}' )

    return row_count, skipped_non_hourly


def parse_datetime_series( series: pd.Series ) -> pd.Series:
    # parse mixed timestamp strings to pandas datetime
    return pd.to_datetime( series, errors = 'coerce' )


def prepare_nutrient_hourly( nutrient_csv: Path ) -> pd.DataFrame:
    # load nutrient file then snap to nearest hour and average per station hour
    print( f'- loading nutrient csv {nutrient_csv}' )
    nutrient_df = pd.read_csv( nutrient_csv, low_memory = False )
    print( f'- nutrient rows loaded {len( nutrient_df )}' )

    nutrient_df[ 'station' ] = nutrient_df[ 'station' ].astype( str ).str.strip( ).str.lower( )
    nutrient_df[ 'region' ] = nutrient_df[ 'region' ].astype( str ).str.strip( ).str.lower( )
    nutrient_df[ 'datetime_parsed' ] = parse_datetime_series( nutrient_df[ 'datetime' ] )
    nutrient_df = nutrient_df.dropna( subset = [ 'datetime_parsed', 'station' ] )

    # nearest hour snap using plus thirty then floor
    nutrient_df[ 'datetime_hour' ] = ( nutrient_df[ 'datetime_parsed' ] + pd.Timedelta( minutes = 30 ) ).dt.floor( 'h' )

    available_cols = [ column for column in NUTRIENT_VALUE_COLUMNS if column in nutrient_df.columns ]
    if not available_cols:
        raise ValueError( 'no nutrient value columns found in nutrient csv' )

    for column in available_cols:
        nutrient_df[ column ] = pd.to_numeric( nutrient_df[ column ], errors = 'coerce' )

    grouped = nutrient_df.groupby( [ 'station', 'datetime_hour' ], as_index = False )[ available_cols ].mean( )
    rename_map = { column: f'nut_{column.lower( )}' for column in available_cols }
    grouped = grouped.rename( columns = rename_map )
    # keep merge key globally monotonic for pandas merge_asof
    grouped = grouped.sort_values( [ 'datetime_hour', 'station' ] ).reset_index( drop = True )

    print( f'- nutrient hourly rows {len( grouped )}' )
    return grouped


def join_wq_with_nutrients_asof( 
    wq_csv: Path,
    nutrient_hourly: pd.DataFrame,
    merged_output: Path,
    *,
    chunk_size: int,
) -> int:
    # chunk through wq then do backward asof merge up to seven days
    if chunk_size <= 0:
        raise ValueError( f'join chunk size must be positive got {chunk_size}' )

    print( f'- loading wq in chunks of {chunk_size}' )
    merged_output.parent.mkdir( parents = True, exist_ok = True )
    if merged_output.exists( ):
        merged_output.unlink( )

    total_rows = 0
    wrote_header = False

    # pandas merge_asof can require the time key to be globally sorted
    nutrient_for_join = nutrient_hourly.sort_values( [ 'datetime_hour', 'station' ], kind = 'mergesort' ).reset_index( drop = True )

    for chunk_index, chunk in enumerate( pd.read_csv( wq_csv, chunksize = chunk_size, low_memory = False ), start = 1 ):
        print( f'- processing wq chunk {chunk_index} rows {len( chunk )}' )

        chunk[ 'station' ] = chunk[ 'station' ].astype( str ).str.strip( ).str.lower( )
        chunk[ 'region' ] = chunk[ 'region' ].astype( str ).str.strip( ).str.lower( )
        chunk[ 'datetime_parsed' ] = parse_datetime_series( chunk[ 'datetime' ] )
        chunk = chunk.dropna( subset = [ 'datetime_parsed', 'station' ] )
        chunk = chunk.sort_values( [ 'datetime_parsed', 'station' ], kind = 'mergesort' )

        merged = pd.merge_asof( 
            chunk,
            nutrient_for_join,
            left_on = 'datetime_parsed',
            right_on = 'datetime_hour',
            by = 'station',
            direction = 'backward',
            tolerance = pd.Timedelta( days = 7 ),
        )

        merged[ 'nut_obs_datetime' ] = merged[ 'datetime_hour' ]
        merged[ 'nut_age_hours' ] = ( merged[ 'datetime_parsed' ] - merged[ 'datetime_hour' ] ).dt.total_seconds( ) / 3600.0

        merged = merged.drop( columns = [ 'datetime_parsed', 'datetime_hour' ], errors = 'ignore' )
        merged.to_csv( merged_output, mode = 'a', index = False, header = not wrote_header )

        wrote_header = True
        total_rows += len( merged )
        print( f'- joined rows written so far {total_rows}' )

    return total_rows


def main( ) -> None:
    args = parse_args( )

    data_dir = args.data_dir.resolve( )
    output_dir = args.output_dir.resolve( )

    if not data_dir.exists( ):
        raise FileNotFoundError( f'missing data directory: {data_dir}' )

    all_files = discover_source_files( data_dir )

    wq_files = [ file_row for file_row in all_files if file_row.file_type == 'wq' ]
    nut_files = [ file_row for file_row in all_files if file_row.file_type == 'nut' ]

    print( f'- discovered {len( all_files )} source files wq {len( wq_files )} nut {len( nut_files )}' )

    if not wq_files:
        raise ValueError( 'no wq files matched naming pattern' )

    if not nut_files:
        raise ValueError( 'no nutrient files matched naming pattern' )

    wq_output = output_dir / WQ_OUTPUT_NAME
    nut_output = output_dir / NUT_OUTPUT_NAME
    merged_output = output_dir / MERGED_OUTPUT_NAME

    print( '- stage 1 collating raw wq and nutrient rows' )
    wq_rows, wq_skipped_non_hourly = write_collated_csv( wq_files, wq_output )
    nut_rows, _ = write_collated_csv( nut_files, nut_output )

    print( f'data dir {data_dir}' )
    print( f'wq files {len( wq_files )} rows {wq_rows} skipped non hourly {wq_skipped_non_hourly} output {wq_output}' )
    print( f'nut files {len( nut_files )} rows {nut_rows} output {nut_output}' )
    print( '- stage 1 complete' )

    # clear large lists before second pass reload
    del all_files
    del wq_files
    del nut_files
    gc.collect( )

    print( '- stage 2 preparing hourly nutrient means and asof join' )
    nutrient_hourly = prepare_nutrient_hourly( nut_output )
    joined_rows = join_wq_with_nutrients_asof( 
        wq_output,
        nutrient_hourly,
        merged_output,
        chunk_size = args.join_chunk_size,
    )

    print( '- stage 2 complete' )
    print( f'joined output rows {joined_rows} output {merged_output}' )



if __name__ == '__main__':
    main( )
