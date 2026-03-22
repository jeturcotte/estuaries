#!/usr/bin/env python3
"""build a local region and station lookup table from NERRS metadata"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path( __file__ ).resolve( ).parents[ 2 ]
DEFAULT_SAMPLING_STATIONS = PROJECT_ROOT / 'Data' / 'NERRS' / 'sampling_stations.csv'
DEFAULT_T4D_WATER_HISTORY = PROJECT_ROOT / 'estuaries' / 'data' / '1hr' / 't4d.1hr.baseline.csv'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'estuaries' / 'data' / 'reference'


def parse_args( ) -> argparse.Namespace:
    parser = argparse.ArgumentParser( description = 'build canonical NERRS station index' )
    parser.add_argument( '--sampling-stations', type = Path, default = DEFAULT_SAMPLING_STATIONS )
    parser.add_argument( '--t4d-water-history', type = Path, default = DEFAULT_T4D_WATER_HISTORY )
    parser.add_argument( '--output-dir', type = Path, default = DEFAULT_OUTPUT_DIR )
    return parser.parse_args( )


def normalize_columns( frame: pd.DataFrame ) -> pd.DataFrame:
    frame = frame.copy( )
    frame.columns = [ column.strip( ) for column in frame.columns ]
    return frame


def apply_longitude_sign( raw_longitude: pd.Series, lat_long_text: pd.Series ) -> pd.Series:
    longitude = pd.to_numeric( raw_longitude, errors = 'coerce' )
    text = lat_long_text.astype( str )

    is_west = text.str.contains( 'W', case = False, na = False )
    is_east = text.str.contains( 'E', case = False, na = False )

    longitude.loc[ is_west ] = -longitude.loc[ is_west ].abs( )
    longitude.loc[ is_east ] = longitude.loc[ is_east ].abs( )

    # fallback is west hemisphere because this NERRS set is US-focused
    unknown_sign = ~( is_west | is_east )
    longitude.loc[ unknown_sign ] = -longitude.loc[ unknown_sign ].abs( )

    return longitude


def pick_mode_table( frame: pd.DataFrame, group_column: str, value_column: str ) -> pd.DataFrame:
    counts = ( 
        frame
        .groupby( [ group_column, value_column ] )
        .size( )
        .reset_index( name = 'n' )
        .sort_values( [ group_column, 'n', value_column ], ascending = [ True, False, True ] )
        .drop_duplicates( subset = [ group_column ] )
    )
    return counts[ [ group_column, value_column ] ]


def load_sampling_station_table( sampling_stations_path: Path ) -> pd.DataFrame:
    raw = pd.read_csv( sampling_stations_path, encoding = 'latin1', low_memory = False )
    raw = normalize_columns( raw )

    required_columns = [ 
        'NERR Site ID',
        'Station Code',
        'Station Name',
        'Reserve Name',
        'Latitude',
        'Longitude',
        'Lat Long',
        'Status',
    ]

    missing_columns = [ column for column in required_columns if column not in raw.columns ]
    if missing_columns:
        raise ValueError( f'missing required columns in sampling stations file: {missing_columns}' )

    table = pd.DataFrame( )
    table[ 'region_code' ] = raw[ 'NERR Site ID' ].astype( str ).str.strip( ).str.lower( )
    table[ 'station_code_full' ] = raw[ 'Station Code' ].astype( str ).str.strip( ).str.lower( )
    table[ 'station' ] = table[ 'station_code_full' ].str[ :5 ]
    table[ 'station_name' ] = raw[ 'Station Name' ].astype( str ).str.strip( )
    table[ 'region_name' ] = raw[ 'Reserve Name' ].astype( str ).str.strip( )
    table[ 'status' ] = raw[ 'Status' ].astype( str ).str.strip( )
    table[ 'latitude' ] = pd.to_numeric( raw[ 'Latitude' ], errors = 'coerce' )
    table[ 'longitude' ] = apply_longitude_sign( raw[ 'Longitude' ], raw[ 'Lat Long' ] )

    table = table.loc[ 
        table[ 'region_code' ].ne( '' ) &
        table[ 'station' ].ne( '' ),
    ].copy( )

    return table


def build_station_index( sampling_station_table: pd.DataFrame ) -> pd.DataFrame:
    table = sampling_station_table.copy( )

    station_names = pick_mode_table( table, group_column = 'station', value_column = 'station_name' )
    station_regions = pick_mode_table( table, group_column = 'station', value_column = 'region_code' )

    region_names = pick_mode_table( table, group_column = 'region_code', value_column = 'region_name' )

    coordinates = ( 
        table
        .groupby( 'station', as_index = False )
        .agg( 
            latitude = ( 'latitude', 'median' ),
            longitude = ( 'longitude', 'median' ),
            n_source_rows = ( 'station_code_full', 'size' ),
            n_unique_station_names = ( 'station_name', pd.Series.nunique ),
            n_unique_coordinate_pairs = ( 'latitude', pd.Series.nunique ),
        )
    )

    station_code_variants = ( 
        table
        .groupby( 'station', as_index = False )
        .agg( station_code_full_variants = ( 'station_code_full', lambda series: '|'.join( sorted( set( series ) ) ) ) )
    )

    index = ( 
        coordinates
        .merge( station_regions, on = 'station', how = 'left' )
        .merge( region_names, on = 'region_code', how = 'left' )
        .merge( station_names, on = 'station', how = 'left' )
        .merge( station_code_variants, on = 'station', how = 'left' )
        .sort_values( [ 'region_code', 'station' ] )
        .reset_index( drop = True )
    )

    index[ 'latitude' ] = index[ 'latitude' ].round( 6 )
    index[ 'longitude' ] = index[ 'longitude' ].round( 6 )

    return index


def load_t4d_keys( t4d_water_history_path: Path ) -> pd.DataFrame:
    if not t4d_water_history_path.exists( ):
        return pd.DataFrame( columns = [ 'region_code', 'station' ] )

    unique_keys: set[ tuple[ str, str ] ] = set( )

    for chunk in pd.read_csv( t4d_water_history_path, usecols = [ 'region', 'station' ], chunksize = 500000, low_memory = False ):
        regions = chunk[ 'region' ].astype( str ).str.strip( ).str.lower( )
        stations = chunk[ 'station' ].astype( str ).str.strip( ).str.lower( )

        unique_keys.update( zip( regions, stations ) )

    key_frame = pd.DataFrame( sorted( unique_keys ), columns = [ 'region_code', 'station' ] )
    return key_frame


def append_t4d_coverage( station_index: pd.DataFrame, t4d_keys: pd.DataFrame ) -> pd.DataFrame:
    index = station_index.copy( )

    if t4d_keys.empty:
        index[ 'in_t4d_1hr_water_history' ] = False
        return index

    merged = index.merge( t4d_keys, on = [ 'region_code', 'station' ], how = 'left', indicator = True )
    merged[ 'in_t4d_1hr_water_history' ] = merged[ '_merge' ].eq( 'both' )
    merged = merged.drop( columns = [ '_merge' ] )
    return merged


def write_outputs( station_index: pd.DataFrame, output_dir: Path ) -> tuple[ Path, Path ]:
    output_dir.mkdir( parents = True, exist_ok = True )

    csv_path = output_dir / 'nerrs_station_index.csv'
    json_path = output_dir / 'nerrs_station_index.json'

    station_index.to_csv( csv_path, index = False )

    records = station_index.to_dict( orient = 'records' )
    with open( json_path, 'w', encoding = 'utf-8' ) as handle:
        json.dump( records, handle, indent = 2 )

    return csv_path, json_path


def main( ) -> None:
    args = parse_args( )

    sampling_station_table = load_sampling_station_table( args.sampling_stations )
    station_index = build_station_index( sampling_station_table )

    t4d_keys = load_t4d_keys( args.t4d_water_history )
    station_index = append_t4d_coverage( station_index, t4d_keys )

    csv_path, json_path = write_outputs( station_index, args.output_dir )

    matched_count = int( station_index[ 'in_t4d_1hr_water_history' ].sum( ) )
    total_count = len( station_index )

    print( f'wrote station index csv: {csv_path}' )
    print( f'wrote station index json: {json_path}' )
    print( f'stations in index: {total_count}' )
    print( f'stations matched in t4d 1hr water history: {matched_count}' )


if __name__ == '__main__':
    main( )
