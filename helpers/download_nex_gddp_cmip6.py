from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from nex_gddp_cmip6 import build_unique_asset_table


DEFAULT_MANIFEST_PATH = Path( 'estuaries/artifacts/nex_gddp_cmip6_station_asset_manifest.csv' )
DEFAULT_OUT_DIR = Path( 'estuaries/data/cmip6' )


def parse_args( ) -> argparse.Namespace:
    parser = argparse.ArgumentParser( 
        description = 'Download NEX-GDDP CMIP6 yearly NetCDF assets from the saved T4D manifest.',
    )
    parser.add_argument( 
        '--manifest-path',
        type = Path,
        default = DEFAULT_MANIFEST_PATH,
        help = 'Path to the station-level asset manifest CSV.',
    )
    parser.add_argument( 
        '--out-dir',
        type = Path,
        default = DEFAULT_OUT_DIR,
        help = 'Directory where downloaded files should be written.',
    )
    parser.add_argument( 
        '--selection-reason',
        nargs = '*',
        default = None,
        help = 'Optional manifest selection_reason values to keep.',
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
        help = 'Optional scenarios to keep, such as historical, ssp245, or ssp585.',
    )
    parser.add_argument( 
        '--variable',
        nargs = '*',
        default = None,
        help = 'Optional variables to keep, such as tas, pr, sfcWind, or rsds.',
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
        '--limit',
        type = int,
        default = None,
        help = 'Optional cap on the number of unique assets to download after filtering.',
    )
    parser.add_argument( 
        '--workers',
        type = int,
        default = 2,
        help = 'Number of parallel download workers.',
    )
    parser.add_argument( 
        '--timeout',
        type = int,
        default = 120,
        help = 'Per-request timeout in seconds.',
    )
    parser.add_argument( 
        '--retries',
        type = int,
        default = 2,
        help = 'Retries per file after the first attempt fails.',
    )
    parser.add_argument( 
        '--force',
        action = 'store_true',
        help = 'Redownload files even when a non-empty local copy already exists.',
    )
    parser.add_argument( 
        '--dry-run',
        action = 'store_true',
        help = 'Show the filtered download set without downloading files.',
    )
    return parser.parse_args( )


def normalize_values( values: Iterable[ str ] | None ) -> set[ str ] | None:
    if values is None:
        return None

    cleaned = { str( value ).strip( ).lower( ) for value in values if str( value ).strip( ) }
    return cleaned if len( cleaned ) > 0 else None


def filter_station_manifest( manifest: pd.DataFrame, args: argparse.Namespace ) -> pd.DataFrame:
    filtered = manifest.copy( )

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
        filtered = filtered.loc[ pd.to_numeric( filtered[ 'year' ], errors = 'coerce' ) >= int( args.year_start ) ]

    if args.year_end is not None and 'year' in filtered.columns:
        filtered = filtered.loc[ pd.to_numeric( filtered[ 'year' ], errors = 'coerce' ) <= int( args.year_end ) ]

    return filtered.reset_index( drop = True )


def summarize_selection( station_manifest: pd.DataFrame, unique_assets: pd.DataFrame ) -> dict:
    station_keys = [ 'region', 'station' ]
    station_count = 0
    if all( col in station_manifest.columns for col in station_keys ):
        station_count = int( station_manifest[ station_keys ].drop_duplicates( ).shape[ 0 ] )

    summary = { 
        'n_station_links': int( len( station_manifest ) ),
        'n_unique_assets': int( len( unique_assets ) ),
        'n_unique_stations': station_count,
        'models': sorted( unique_assets[ 'model' ].dropna( ).astype( str ).unique( ).tolist( ) ) if 'model' in unique_assets.columns else [ ],
        'scenarios': sorted( unique_assets[ 'scenario' ].dropna( ).astype( str ).unique( ).tolist( ) ) if 'scenario' in unique_assets.columns else [ ],
        'variables': sorted( unique_assets[ 'variable' ].dropna( ).astype( str ).unique( ).tolist( ) ) if 'variable' in unique_assets.columns else [ ],
        'year_start': int( pd.to_numeric( unique_assets[ 'year' ], errors = 'coerce' ).min( ) ) if len( unique_assets ) > 0 and 'year' in unique_assets.columns else None,
        'year_end': int( pd.to_numeric( unique_assets[ 'year' ], errors = 'coerce' ).max( ) ) if len( unique_assets ) > 0 and 'year' in unique_assets.columns else None,
    }

    if 'selection_reason' in station_manifest.columns:
        selection_counts = ( 
            station_manifest[ [ 'region', 'station', 'selection_reason' ] ]
            .drop_duplicates( )
            .groupby( 'selection_reason' )
            .size( )
            .sort_index( )
        )
        summary[ 'selection_reason_counts' ] = { str( key ): int( value ) for key, value in selection_counts.items( ) }

    return summary


def save_selection_metadata( 
    station_manifest: pd.DataFrame,
    unique_assets: pd.DataFrame,
    summary: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir( parents = True, exist_ok = True )
    station_manifest.to_csv( out_dir / 'download_station_asset_manifest.csv', index = False )
    unique_assets.to_csv( out_dir / 'download_unique_assets.csv', index = False )
    ( out_dir / 'download_summary.json' ).write_text( json.dumps( summary, indent = 2 ) )


def build_local_path( out_dir: Path, relative_path: str ) -> Path:
    return out_dir / Path( str( relative_path ) )


def download_asset_row( 
    row: pd.Series,
    out_dir: Path,
    timeout: int,
    retries: int,
    force: bool,
) -> dict:
    relative_path = str( row[ 'relative_path' ] )
    file_url = str( row[ 'file_url' ] )
    local_path = build_local_path( out_dir, relative_path )
    local_path.parent.mkdir( parents = True, exist_ok = True )

    if local_path.exists( ) and local_path.stat( ).st_size > 0 and not force:
        return { 
            'status': 'skipped',
            'relative_path': relative_path,
            'local_path': str( local_path ),
            'bytes': int( local_path.stat( ).st_size ),
        }

    temp_path = local_path.with_suffix( local_path.suffix + '.part' )
    attempts = int( retries ) + 1

    for attempt_idx in range( attempts ):
        try:
            request = Request( 
                file_url,
                headers = { 
                    'User-Agent': 't4d-nex-gddp-cmip6-downloader/1.0',
                },
            )
            with urlopen( request, timeout = timeout ) as response, temp_path.open( 'wb' ) as out_file:
                shutil.copyfileobj( response, out_file, length = 1024 * 1024 )

            temp_path.replace( local_path )

            return { 
                'status': 'downloaded',
                'relative_path': relative_path,
                'local_path': str( local_path ),
                'bytes': int( local_path.stat( ).st_size ),
            }

        except ( HTTPError, URLError, TimeoutError, OSError ) as exc:
            if temp_path.exists( ):
                temp_path.unlink( )

            if attempt_idx >= attempts - 1:
                return { 
                    'status': 'failed',
                    'relative_path': relative_path,
                    'local_path': str( local_path ),
                    'error': f'{ type( exc ).__name__ }: { exc }',
                }

            time.sleep( 1.0 + attempt_idx )

    return { 
        'status': 'failed',
        'relative_path': relative_path,
        'local_path': str( local_path ),
        'error': 'unreachable retry state',
    }


def download_assets( 
    unique_assets: pd.DataFrame,
    out_dir: Path,
    timeout: int,
    retries: int,
    force: bool,
    workers: int,
) -> list[ dict ]:
    rows = [ row for _, row in unique_assets.iterrows( ) ]
    results: list[ dict ] = [ ]

    with ThreadPoolExecutor( max_workers = max( 1, int( workers ) ) ) as executor:
        futures = { 
            executor.submit( 
                download_asset_row,
                row,
                out_dir,
                timeout,
                retries,
                force,
            ): row
            for row in rows
        }

        for future in as_completed( futures ):
            result = future.result( )
            results.append( result )

            status = result.get( 'status' )
            relative_path = result.get( 'relative_path' )
            if status == 'downloaded':
                print( f'downloaded: { relative_path }' )

            elif status == 'skipped':
                print( f'skipped: { relative_path }' )

            else:
                print( f'failed: { relative_path } :: { result.get( "error" ) }', file = sys.stderr )

    return results


def main( ) -> int:
    args = parse_args( )

    manifest_path = Path( args.manifest_path )
    if not manifest_path.exists( ):
        print( f'manifest not found: { manifest_path }', file = sys.stderr )
        return 2

    station_manifest = pd.read_csv( manifest_path )
    filtered_station_manifest = filter_station_manifest( station_manifest, args )

    if len( filtered_station_manifest ) == 0:
        print( 'no manifest rows remain after filtering', file = sys.stderr )
        return 1

    unique_assets = build_unique_asset_table( filtered_station_manifest )
    if args.limit is not None:
        unique_assets = unique_assets.head( int( args.limit ) ).reset_index( drop = True )
        keep_paths = set( unique_assets[ 'relative_path' ].astype( str ) )
        filtered_station_manifest = filtered_station_manifest.loc[ 
            filtered_station_manifest[ 'relative_path' ].astype( str ).isin( keep_paths )
        ].reset_index( drop = True )

    summary = summarize_selection( filtered_station_manifest, unique_assets )
    save_selection_metadata( filtered_station_manifest, unique_assets, summary, Path( args.out_dir ) )

    print( json.dumps( summary, indent = 2 ) )

    if args.dry_run:
        return 0

    results = download_assets( 
        unique_assets,
        Path( args.out_dir ),
        args.timeout,
        args.retries,
        args.force,
        args.workers,
    )

    downloaded = sum( 1 for item in results if item.get( 'status' ) == 'downloaded' )
    skipped = sum( 1 for item in results if item.get( 'status' ) == 'skipped' )
    failed = [ item for item in results if item.get( 'status' ) == 'failed' ]
    bytes_on_disk = int( sum( int( item.get( 'bytes', 0 ) ) for item in results if item.get( 'status' ) in [ 'downloaded', 'skipped' ] ) )

    final_summary = { 
        **summary,
        'downloaded': downloaded,
        'skipped': skipped,
        'failed': len( failed ),
        'bytes_on_disk': bytes_on_disk,
    }
    ( Path( args.out_dir ) / 'download_results.json' ).write_text( json.dumps( final_summary, indent = 2 ) )

    print( json.dumps( final_summary, indent = 2 ) )

    if len( failed ) > 0:
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit( main( ) )
