from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nex_gddp_cmip6 import DEFAULT_NEX_VARIABLE_SPECS

try:
    from netCDF4 import Dataset, num2date

except ImportError as exc:
    raise SystemExit( 
        'netCDF4 is required for this script. Run it with the project virtualenv, for example '
        '`./.venv/bin/python estuaries/helpers/extract_nex_gddp_cmip6_station_series.py ...`.'
    ) from exc


DEFAULT_STATION_TARGET_PATH = Path( 'estuaries/artifacts/nex_gddp_cmip6_station_targets.csv' )
DEFAULT_ASSET_TABLE_PATH = Path( 'estuaries/artifacts/nex_gddp_cmip6_unique_assets.csv' )
DEFAULT_CMIP6_ROOT = Path( 'estuaries/data/cmip6' )
DEFAULT_OUT_DIR = Path( 'estuaries/data/cmip6_station_series' )


def parse_args( ) -> argparse.Namespace:
    parser = argparse.ArgumentParser( 
        description = 'Extract nearest-cell NEX-GDDP CMIP6 daily station time series from downloaded yearly NetCDF files.',
    )
    parser.add_argument( 
        '--station-target-path',
        type = Path,
        default = DEFAULT_STATION_TARGET_PATH,
        help = 'Path to the station target CSV created by the Phase 9 manifest step.',
    )
    parser.add_argument( 
        '--asset-table-path',
        type = Path,
        default = DEFAULT_ASSET_TABLE_PATH,
        help = 'Path to the unique asset CSV created by the Phase 9 manifest step.',
    )
    parser.add_argument( 
        '--cmip6-root',
        type = Path,
        default = DEFAULT_CMIP6_ROOT,
        help = 'Root directory that contains the downloaded CMIP6 files.',
    )
    parser.add_argument( 
        '--out-dir',
        type = Path,
        default = DEFAULT_OUT_DIR,
        help = 'Directory where extracted station series should be written.',
    )
    parser.add_argument( 
        '--selection-reason',
        nargs = '*',
        default = None,
        help = 'Optional station selection_reason values to keep.',
    )
    parser.add_argument( 
        '--region',
        nargs = '*',
        default = None,
        help = 'Optional region codes to keep.',
    )
    parser.add_argument( 
        '--station',
        nargs = '*',
        default = None,
        help = 'Optional station codes to keep.',
    )
    parser.add_argument( 
        '--model',
        nargs = '*',
        default = None,
        help = 'Optional CMIP6 model names to keep.',
    )
    parser.add_argument( 
        '--scenario',
        nargs = '*',
        default = None,
        help = 'Optional scenarios to keep.',
    )
    parser.add_argument( 
        '--variable',
        nargs = '*',
        default = None,
        help = 'Optional CMIP6 variables to keep.',
    )
    parser.add_argument( 
        '--year-start',
        type = int,
        default = None,
        help = 'Optional first year to keep.',
    )
    parser.add_argument( 
        '--year-end',
        type = int,
        default = None,
        help = 'Optional last year to keep.',
    )
    parser.add_argument( 
        '--limit-assets',
        type = int,
        default = None,
        help = 'Optional cap on the number of local yearly assets to process after filtering.',
    )
    parser.add_argument( 
        '--overwrite',
        action = 'store_true',
        help = 'Replace existing output files instead of merging with them.',
    )
    parser.add_argument( 
        '--dry-run',
        action = 'store_true',
        help = 'Show the filtered station and asset selection without extracting series.',
    )
    return parser.parse_args( )


def normalize_values( values: list[ str ] | None ) -> set[ str ] | None:
    if values is None:
        return None

    cleaned = { str( value ).strip( ).lower( ) for value in values if str( value ).strip( ) }
    return cleaned if len( cleaned ) > 0 else None


def filter_station_frame( station_frame: pd.DataFrame, args: argparse.Namespace ) -> pd.DataFrame:
    filtered = station_frame.copy( )

    selection_reason_filter = normalize_values( args.selection_reason )
    if selection_reason_filter is not None and 'selection_reason' in filtered.columns:
        filtered = filtered.loc[ 
            filtered[ 'selection_reason' ].astype( str ).str.lower( ).isin( selection_reason_filter )
        ]

    region_filter = normalize_values( args.region )
    if region_filter is not None and 'region' in filtered.columns:
        filtered = filtered.loc[ 
            filtered[ 'region' ].astype( str ).str.lower( ).isin( region_filter )
        ]

    station_filter = normalize_values( args.station )
    if station_filter is not None and 'station' in filtered.columns:
        filtered = filtered.loc[ 
            filtered[ 'station' ].astype( str ).str.lower( ).isin( station_filter )
        ]

    return filtered.reset_index( drop = True )


def filter_asset_frame( asset_frame: pd.DataFrame, args: argparse.Namespace ) -> pd.DataFrame:
    filtered = asset_frame.copy( )

    model_filter = normalize_values( args.model )
    if model_filter is not None and 'model' in filtered.columns:
        filtered = filtered.loc[ 
            filtered[ 'model' ].astype( str ).str.lower( ).isin( model_filter )
        ]

    scenario_filter = normalize_values( args.scenario )
    if scenario_filter is not None and 'scenario' in filtered.columns:
        filtered = filtered.loc[ 
            filtered[ 'scenario' ].astype( str ).str.lower( ).isin( scenario_filter )
        ]

    variable_filter = normalize_values( args.variable )
    if variable_filter is not None and 'variable' in filtered.columns:
        filtered = filtered.loc[ 
            filtered[ 'variable' ].astype( str ).str.lower( ).isin( variable_filter )
        ]

    if args.year_start is not None and 'year' in filtered.columns:
        filtered = filtered.loc[ 
            pd.to_numeric( filtered[ 'year' ], errors = 'coerce' ) >= int( args.year_start )
        ]

    if args.year_end is not None and 'year' in filtered.columns:
        filtered = filtered.loc[ 
            pd.to_numeric( filtered[ 'year' ], errors = 'coerce' ) <= int( args.year_end )
        ]

    return filtered.reset_index( drop = True )


def lon_to_360( lon_value: float ) -> float:
    lon_360 = float( lon_value ) % 360.0
    return lon_360 if lon_360 >= 0.0 else lon_360 + 360.0


def angular_lon_diff( lon_values: np.ndarray, station_lon_360: float ) -> np.ndarray:
    lon_diff = np.abs( lon_values - station_lon_360 )
    return np.minimum( lon_diff, 360.0 - lon_diff )


def build_station_lookup_for_asset( 
    sample_path: Path,
    variable_name: str,
    station_frame: pd.DataFrame,
) -> pd.DataFrame:
    with Dataset( str( sample_path ) ) as ds:
        lat_values = np.asarray( ds.variables[ 'lat' ][ : ], dtype = 'float64' )
        lon_values = np.asarray( ds.variables[ 'lon' ][ : ], dtype = 'float64' )
        first_slice = np.ma.filled( ds.variables[ variable_name ][ 0, :, : ], np.nan )

    valid_mask = np.isfinite( first_slice )
    lat_grid, lon_grid = np.meshgrid( lat_values, lon_values, indexing = 'ij' )
    valid_indices = np.argwhere( valid_mask )
    valid_lats = lat_grid[ valid_mask ]
    valid_lons = lon_grid[ valid_mask ]

    if len( valid_indices ) == 0:
        full_indices = np.indices( first_slice.shape )
        valid_indices = np.column_stack( [ full_indices[ 0 ].ravel( ), full_indices[ 1 ].ravel( ) ] )
        valid_lats = lat_grid.ravel( )
        valid_lons = lon_grid.ravel( )

    lookup_rows = [ ]

    for station_row in station_frame.itertuples( index = False ):
        station_lat = float( station_row.latitude )
        station_lon = float( station_row.longitude )
        station_lon_360 = lon_to_360( station_lon )

        lat_diff = valid_lats - station_lat
        lon_diff = angular_lon_diff( valid_lons, station_lon_360 )
        distance_deg = np.sqrt( lat_diff ** 2 + lon_diff ** 2 )
        best_idx = int( np.argmin( distance_deg ) )
        lat_idx, lon_idx = valid_indices[ best_idx ]

        lookup_rows.append( 
            { 
                'region': station_row.region,
                'station': station_row.station,
                'selection_reason': getattr( station_row, 'selection_reason', None ),
                'station_latitude': station_lat,
                'station_longitude': station_lon,
                'lat_index': int( lat_idx ),
                'lon_index': int( lon_idx ),
                'grid_latitude': float( lat_values[ lat_idx ] ),
                'grid_longitude_360': float( lon_values[ lon_idx ] ),
                'grid_longitude': float( lon_values[ lon_idx ] - 360.0 if lon_values[ lon_idx ] > 180.0 else lon_values[ lon_idx ] ),
                'grid_distance_deg': float( distance_deg[ best_idx ] ),
                'lookup_source_file': str( sample_path ),
            }
        )

    return pd.DataFrame( lookup_rows )


def convert_target_values( variable_name: str, native_values: np.ndarray ) -> np.ndarray:
    values = np.asarray( native_values, dtype = 'float64' )

    if variable_name == 'tas':
        return values - 273.15

    if variable_name == 'pr':
        return values * 86400.0

    return values


def convert_time_values( time_var ) -> pd.Series:
    calendar = getattr( time_var, 'calendar', 'standard' )
    datetimes = num2date( 
        time_var[ : ],
        units = time_var.units,
        calendar = calendar,
        only_use_cftime_datetimes = False,
        only_use_python_datetimes = False,
    )
    date_text = [ f'{ item.year:04d}-{ item.month:02d}-{ item.day:02d}' for item in datetimes ]
    return pd.Series( date_text, dtype = 'string' )


def station_output_path( 
    out_dir: Path,
    region: str,
    station: str,
    model_name: str,
    scenario_name: str,
    variable_name: str,
) -> Path:
    return ( 
        out_dir
        / 'station_series'
        / f'{ region }_{ station }'
        / model_name
        / scenario_name
        / f'{ variable_name }.csv'
    )


def merge_and_write_csv( 
    frame: pd.DataFrame,
    out_path: Path,
    dedupe_cols: list[ str ],
    overwrite: bool,
    allow_replace_existing: bool,
) -> None:
    out_path.parent.mkdir( parents = True, exist_ok = True )

    if out_path.exists( ) and not ( overwrite and allow_replace_existing ):
        existing = pd.read_csv( out_path )
        combined = pd.concat( [ existing, frame ], ignore_index = True, sort = False )

    else:
        combined = frame.copy( )

    combined = combined.drop_duplicates( subset = dedupe_cols ).sort_values( dedupe_cols ).reset_index( drop = True )
    combined.to_csv( out_path, index = False )


def summarize_selection( 
    station_frame: pd.DataFrame,
    asset_frame: pd.DataFrame,
    downloaded_asset_frame: pd.DataFrame,
) -> dict:
    summary = { 
        'n_selected_stations': int( station_frame[ [ 'region', 'station' ] ].drop_duplicates( ).shape[ 0 ] ),
        'n_requested_assets': int( len( asset_frame ) ),
        'n_local_assets': int( len( downloaded_asset_frame ) ),
        'models': sorted( downloaded_asset_frame[ 'model' ].dropna( ).astype( str ).unique( ).tolist( ) ) if 'model' in downloaded_asset_frame.columns else [ ],
        'scenarios': sorted( downloaded_asset_frame[ 'scenario' ].dropna( ).astype( str ).unique( ).tolist( ) ) if 'scenario' in downloaded_asset_frame.columns else [ ],
        'variables': sorted( downloaded_asset_frame[ 'variable' ].dropna( ).astype( str ).unique( ).tolist( ) ) if 'variable' in downloaded_asset_frame.columns else [ ],
        'year_start': int( pd.to_numeric( downloaded_asset_frame[ 'year' ], errors = 'coerce' ).min( ) ) if len( downloaded_asset_frame ) > 0 else None,
        'year_end': int( pd.to_numeric( downloaded_asset_frame[ 'year' ], errors = 'coerce' ).max( ) ) if len( downloaded_asset_frame ) > 0 else None,
    }

    if 'selection_reason' in station_frame.columns:
        counts = ( 
            station_frame[ [ 'region', 'station', 'selection_reason' ] ]
            .drop_duplicates( )
            .groupby( 'selection_reason' )
            .size( )
            .sort_index( )
        )
        summary[ 'selection_reason_counts' ] = { str( key ): int( value ) for key, value in counts.items( ) }

    return summary


def build_station_series_frame( 
    ds,
    asset_row: pd.Series,
    station_row: pd.Series,
    lookup_row: pd.Series,
) -> pd.DataFrame:
    variable_name = str( asset_row[ 'variable' ] )
    native_units = getattr( ds.variables[ variable_name ], 'units', asset_row.get( 'native_units', None ) )
    target_units = DEFAULT_NEX_VARIABLE_SPECS.get( variable_name, { } ).get( 'target_units', native_units )

    date_series = convert_time_values( ds.variables[ 'time' ] )
    native_values = np.ma.filled( 
        ds.variables[ variable_name ][ :, int( lookup_row[ 'lat_index' ] ), int( lookup_row[ 'lon_index' ] ) ],
        np.nan,
    )
    target_values = convert_target_values( variable_name, native_values )

    return pd.DataFrame( 
        { 
            'date': date_series,
            'year': int( asset_row[ 'year' ] ),
            'region': station_row[ 'region' ],
            'station': station_row[ 'station' ],
            'selection_reason': station_row.get( 'selection_reason', None ),
            'latitude': station_row[ 'latitude' ],
            'longitude': station_row[ 'longitude' ],
            'model': asset_row[ 'model' ],
            'scenario': asset_row[ 'scenario' ],
            'variable': variable_name,
            'target_driver': DEFAULT_NEX_VARIABLE_SPECS.get( variable_name, { } ).get( 'target_driver', None ),
            'native_units': native_units,
            'target_units': target_units,
            'value_native': native_values.astype( 'float64' ),
            'value_target': target_values.astype( 'float64' ),
            'grid_latitude': lookup_row[ 'grid_latitude' ],
            'grid_longitude': lookup_row[ 'grid_longitude' ],
            'grid_distance_deg': lookup_row[ 'grid_distance_deg' ],
            'source_relative_path': asset_row[ 'relative_path' ],
        }
    )


def aggregate_station_year_frame( station_year_frame: pd.DataFrame, group_col: str ) -> pd.DataFrame:
    aggregate = ( 
        station_year_frame
        .groupby( 
            [ 'date', 'model', 'scenario', 'variable', group_col ],
            as_index = False,
        )
        .agg( 
            n_stations = ( 'station', 'nunique' ),
            value_native_mean = ( 'value_native', 'mean' ),
            value_target_mean = ( 'value_target', 'mean' ),
        )
    )
    return aggregate


def main( ) -> int:
    args = parse_args( )

    station_target_path = Path( args.station_target_path )
    asset_table_path = Path( args.asset_table_path )
    cmip6_root = Path( args.cmip6_root )
    out_dir = Path( args.out_dir )

    if not station_target_path.exists( ):
        print( f'station target file not found: { station_target_path }', file = sys.stderr )
        return 2

    if not asset_table_path.exists( ):
        print( f'asset table not found: { asset_table_path }', file = sys.stderr )
        return 2

    station_frame = pd.read_csv( station_target_path )
    station_frame = filter_station_frame( station_frame, args )
    if len( station_frame ) == 0:
        print( 'no stations remain after filtering', file = sys.stderr )
        return 1

    asset_frame = pd.read_csv( asset_table_path )
    asset_frame = filter_asset_frame( asset_frame, args )
    if len( asset_frame ) == 0:
        print( 'no assets remain after filtering', file = sys.stderr )
        return 1

    asset_frame[ 'local_path' ] = asset_frame[ 'relative_path' ].apply( lambda value: str( cmip6_root / Path( str( value ) ) ) )
    asset_frame[ 'is_downloaded' ] = asset_frame[ 'local_path' ].apply( lambda value: Path( value ).exists( ) )
    downloaded_asset_frame = asset_frame.loc[ asset_frame[ 'is_downloaded' ] ].copy( ).reset_index( drop = True )

    if args.limit_assets is not None:
        downloaded_asset_frame = downloaded_asset_frame.head( int( args.limit_assets ) ).reset_index( drop = True )

    if len( downloaded_asset_frame ) == 0:
        print( 'no local CMIP6 files matched the current filters', file = sys.stderr )
        return 1

    summary = summarize_selection( station_frame, asset_frame, downloaded_asset_frame )
    print( json.dumps( summary, indent = 2 ) )

    out_dir.mkdir( parents = True, exist_ok = True )
    station_frame.to_csv( out_dir / 'selected_station_targets.csv', index = False )
    downloaded_asset_frame.to_csv( out_dir / 'selected_local_assets.csv', index = False )
    ( out_dir / 'selection_summary.json' ).write_text( json.dumps( summary, indent = 2 ) )

    if args.dry_run:
        return 0

    lookup_cache: dict[ tuple[ str, str, str ], pd.DataFrame ] = { }
    lookup_rows = [ ]
    processed_assets = 0
    written_station_files: set[ str ] = set( )
    initialized_output_paths: set[ str ] = set( )

    selection_aggregate_path = out_dir / 'selection_reason_daily_aggregate.csv'
    region_aggregate_path = out_dir / 'region_daily_aggregate.csv'

    for asset_row in downloaded_asset_frame.sort_values( [ 'model', 'scenario', 'variable', 'year' ] ).itertuples( index = False ):
        asset_series = pd.Series( asset_row._asdict( ) )
        key = ( str( asset_series[ 'model' ] ), str( asset_series[ 'scenario' ] ), str( asset_series[ 'variable' ] ) )

        if key not in lookup_cache:
            lookup_cache[ key ] = build_station_lookup_for_asset( 
                Path( str( asset_series[ 'local_path' ] ) ),
                str( asset_series[ 'variable' ] ),
                station_frame,
            )
            lookup_block = lookup_cache[ key ].copy( )
            lookup_block[ 'model' ] = key[ 0 ]
            lookup_block[ 'scenario' ] = key[ 1 ]
            lookup_block[ 'variable' ] = key[ 2 ]
            lookup_rows.append( lookup_block )

        key_lookup = lookup_cache[ key ]
        station_year_frames = [ ]

        with Dataset( str( asset_series[ 'local_path' ] ) ) as ds:
            for station_row in station_frame.itertuples( index = False ):
                station_series = pd.Series( station_row._asdict( ) )
                lookup_row = key_lookup.loc[ 
                    ( key_lookup[ 'region' ] == station_series[ 'region' ] )
                    & ( key_lookup[ 'station' ] == station_series[ 'station' ] )
                ]

                if len( lookup_row ) == 0:
                    continue

                station_year_frame = build_station_series_frame( 
                    ds,
                    asset_series,
                    station_series,
                    lookup_row.iloc[ 0 ],
                )
                station_year_frames.append( station_year_frame )

                station_file_path = station_output_path( 
                    out_dir,
                    str( station_series[ 'region' ] ),
                    str( station_series[ 'station' ] ),
                    str( asset_series[ 'model' ] ),
                    str( asset_series[ 'scenario' ] ),
                    str( asset_series[ 'variable' ] ),
                )
                merge_and_write_csv( 
                    station_year_frame,
                    station_file_path,
                    dedupe_cols = [ 'date', 'model', 'scenario', 'variable' ],
                    overwrite = args.overwrite,
                    allow_replace_existing = str( station_file_path ) not in initialized_output_paths,
                )
                written_station_files.add( str( station_file_path ) )
                initialized_output_paths.add( str( station_file_path ) )

        if len( station_year_frames ) == 0:
            continue

        combined_station_year = pd.concat( station_year_frames, ignore_index = True, sort = False )

        if 'selection_reason' in combined_station_year.columns:
            selection_aggregate = aggregate_station_year_frame( combined_station_year, 'selection_reason' )
            merge_and_write_csv( 
                selection_aggregate,
                selection_aggregate_path,
                dedupe_cols = [ 'date', 'model', 'scenario', 'variable', 'selection_reason' ],
                overwrite = args.overwrite,
                allow_replace_existing = str( selection_aggregate_path ) not in initialized_output_paths,
            )
            initialized_output_paths.add( str( selection_aggregate_path ) )

        region_aggregate = aggregate_station_year_frame( combined_station_year, 'region' )
        merge_and_write_csv( 
            region_aggregate,
            region_aggregate_path,
            dedupe_cols = [ 'date', 'model', 'scenario', 'variable', 'region' ],
            overwrite = args.overwrite,
            allow_replace_existing = str( region_aggregate_path ) not in initialized_output_paths,
        )
        initialized_output_paths.add( str( region_aggregate_path ) )

        processed_assets += 1
        print( f'processed: { asset_series[ "relative_path" ] }' )

    if len( lookup_rows ) > 0:
        lookup_frame = pd.concat( lookup_rows, ignore_index = True, sort = False )
        lookup_frame = lookup_frame.drop_duplicates( 
            subset = [ 'region', 'station', 'model', 'scenario', 'variable' ]
        ).sort_values( [ 'model', 'scenario', 'variable', 'region', 'station' ] )
        lookup_frame.to_csv( out_dir / 'station_grid_lookup.csv', index = False )

    final_summary = { 
        **summary,
        'processed_assets': int( processed_assets ),
        'written_station_files': int( len( written_station_files ) ),
        'lookup_keys': int( len( lookup_cache ) ),
        'selection_aggregate_path': str( selection_aggregate_path.resolve( ) ),
        'region_aggregate_path': str( region_aggregate_path.resolve( ) ),
    }
    ( out_dir / 'extraction_summary.json' ).write_text( json.dumps( final_summary, indent = 2 ) )
    print( json.dumps( final_summary, indent = 2 ) )

    return 0


if __name__ == '__main__':
    raise SystemExit( main( ) )
