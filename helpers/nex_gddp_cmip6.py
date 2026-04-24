from pathlib import Path

import pandas as pd


NEX_GDDP_BASE_URL = 'https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com'
NEX_GDDP_DATA_PREFIX = 'NEX-GDDP-CMIP6'

DEFAULT_NEX_MODEL_VARIANTS = { 
    'ACCESS-CM2': 'r1i1p1f1',
    'ACCESS-ESM1-5': 'r1i1p1f1',
    'BCC-CSM2-MR': 'r1i1p1f1',
    'CanESM5': 'r1i1p1f1',
    'MRI-ESM2-0': 'r1i1p1f1',
}

DEFAULT_NEX_MODELS = [ 
    'ACCESS-CM2',
    'ACCESS-ESM1-5',
    'BCC-CSM2-MR',
    'CanESM5',
    'MRI-ESM2-0',
]

DEFAULT_NEX_VARIABLE_SPECS = { 
    'tas': { 
        'target_driver': 'air_temp',
        'native_units': 'K',
        'target_units': 'C',
        'conversion_note': 'target_c = tas - 273.15',
    },
    'pr': { 
        'target_driver': 'precip',
        'native_units': 'kg m-2 s-1',
        'target_units': 'mm day-1',
        'conversion_note': 'target_mm_day = pr * 86400',
    },
    'sfcWind': { 
        'target_driver': 'wind_speed',
        'native_units': 'm s-1',
        'target_units': 'm s-1',
        'conversion_note': 'no unit conversion',
    },
    'rsds': { 
        'target_driver': 'solar',
        'native_units': 'W m-2',
        'target_units': 'W m-2',
        'conversion_note': 'no unit conversion',
    },
}

DEFAULT_NEX_SCENARIO_YEARS = { 
    'historical': ( 1950, 2014 ),
    'ssp245': ( 2015, 2100 ),
    'ssp585': ( 2015, 2100 ),
}


def _first_available( frame, candidates, fill_value = None ):
    for col in candidates:
        if col in frame.columns:
            return frame[ col ]

    return pd.Series( fill_value, index = frame.index )


def select_priority_wobbling_station( first_event, station_baseline_display, exclude_regions = None ):
    """ 
    Select a priority station for wobbling based on the first event frame 
    and station baseline display, with optional exclusion of certain regions.
    """
    exclude_regions = { str( code ).strip( ).lower( ) for code in ( exclude_regions or [ ] ) }

    wobble = first_event.copy( )
    wobble = wobble.merge( station_baseline_display.copy( ), on = [ 'region', 'station' ], how = 'left' )
    wobble = wobble.loc[ wobble[ 'event_date' ].notna( ) ].copy( ) if 'event_date' in wobble.columns else wobble.copy( )
    if len( exclude_regions ) > 0:
        wobble = wobble.loc[ ~wobble[ 'region' ].astype( str ).str.lower( ).isin( exclude_regions ) ].copy( )

    wobble[ 'n_years_present_rank' ] = pd.to_numeric( _first_available( wobble, [ 'n_years_present' ], 0 ), errors = 'coerce' ).fillna( 0.0 )
    wobble[ 'mean_year_coverage_rank' ] = pd.to_numeric( _first_available( wobble, [ 'mean_year_coverage' ], 0 ), errors = 'coerce' ).fillna( 0.0 )
    wobble[ 'n_obs_total_rank' ] = pd.to_numeric( _first_available( wobble, [ 'n_obs_total' ], 0 ), errors = 'coerce' ).fillna( 0.0 )
    wobble[ 'is_partial_baseline_rank' ] = _first_available( wobble, [ 'is_partial_baseline' ], False ).fillna( False ).astype( bool )

    wobble = wobble.sort_values( [ 
        'is_partial_baseline_rank',
        'mean_year_coverage_rank',
        'n_years_present_rank',
        'n_obs_total_rank',
        'event_date',
        'region',
        'station',
    ], ascending = [ True, False, False, False, True, True, True ] ).reset_index( drop = True )

    return wobble.head( 1 ).copy( ), wobble


def build_priority_station_frame( 
    station_baseline_display,
    first_event,
    chesapeake_region_codes = None,
    include_extra_wobbling = True,
):
    """ Build a priority station frame for the Chesapeake region, with optional inclusion of extra wobbling candidates. """
    # see t4d baseline data
    chesapeake_region_codes = chesapeake_region_codes or [ 'cbm', 'cbv' ]
    chesapeake_region_codes_norm = { str( code ).strip( ).lower( ) for code in chesapeake_region_codes }

    # start with all stations in the chesapeake region, then add the top wobbling 
    # candidate if it's not already included in that set
    station_frame = station_baseline_display.copy( )
    station_frame = station_frame.drop_duplicates( subset = [ 'region', 'station' ] ).reset_index( drop = True )
    station_frame = station_frame.loc[ station_frame[ 'region' ].astype( str ).str.lower( ).isin( chesapeake_region_codes_norm ) ].copy( )
    station_frame[ 'selection_reason' ] = 'chesapeake'

    wobble_top, wobble_ranked = select_priority_wobbling_station( 
        first_event,
        station_baseline_display,
        exclude_regions = chesapeake_region_codes,
    )
    if include_extra_wobbling and len( wobble_top ) > 0:
        wobble_top = wobble_top.copy( )
        wobble_top[ 'selection_reason' ] = 'extra_wobbling_complete'
        station_frame = pd.concat( [ station_frame, wobble_top[ station_frame.columns ] ], ignore_index = True, sort = False )
        station_frame = station_frame.drop_duplicates( subset = [ 'region', 'station' ] ).reset_index( drop = True )

    return station_frame, wobble_ranked


def build_station_bbox( station_frame, padding_deg = 0.25 ):
    """ cmip6 stuff is on a 0.25 degree grid, so add a little padding to ensure we capture all relevant cells """
    # we build a bbox for the station frame to help guide spatial subsetting of the cmip6 data ...
    bbox = { 
        'lon_min': float( pd.to_numeric( station_frame[ 'longitude' ], errors = 'coerce' ).min( ) - padding_deg ),
        'lon_max': float( pd.to_numeric( station_frame[ 'longitude' ], errors = 'coerce' ).max( ) + padding_deg ),
        'lat_min': float( pd.to_numeric( station_frame[ 'latitude' ], errors = 'coerce' ).min( ) - padding_deg ),
        'lat_max': float( pd.to_numeric( station_frame[ 'latitude' ], errors = 'coerce' ).max( ) + padding_deg ),
    }
    return bbox


def build_nex_asset_manifest( 
    station_frame,
    models = None,
    model_variants = None,
    variables = None,
    variable_specs = None,
    scenario_years = None,
    base_url = NEX_GDDP_BASE_URL,
    data_prefix = NEX_GDDP_DATA_PREFIX,
):
    """ Build a manifest of NEX-GDDP CMIP6 assets for the given station frame and parameters. """
    models = models or DEFAULT_NEX_MODELS
    model_variants = model_variants or DEFAULT_NEX_MODEL_VARIANTS
    variable_specs = variable_specs or DEFAULT_NEX_VARIABLE_SPECS
    variables = variables or list( variable_specs.keys( ) )
    scenario_years = scenario_years or DEFAULT_NEX_SCENARIO_YEARS

    station_frame = station_frame.copy( )
    station_frame = station_frame.drop_duplicates( subset = [ 'region', 'station' ] ).reset_index( drop = True )

    rows = [ ]
    # we build the manifest with one row per station-model-scenario-variable-year combination
    # even though many of these will point to the same file assets, 
    # because this format is more directly useful for downstream processing and analysis; 
    # we can always collapse to unique assets later if needed
    for station_row in station_frame.itertuples( index = False ):
        # files are organized by model / scenario / member / variable / year,
        for model_name in models:
            member_id = model_variants[ model_name ]
            # and the same files are used for all stations in the same region, 
            # so we don't need to differentiate by station for the file paths
            for scenario_name, ( year_start, year_end ) in scenario_years.items( ):
                for variable_name in variables:
                    spec = variable_specs[ variable_name ]
                    for year in range( int( year_start ), int( year_end ) + 1 ):
                        filename = f'{ variable_name }_day_{ model_name }_{ scenario_name }_{ member_id }_gn_{ year }.nc'
                        relative_path = f'{ data_prefix }/{ model_name }/{ scenario_name }/{ member_id }/{ variable_name }/{ filename }'
                        rows.append( { 
                            'region': station_row.region,
                            'station': station_row.station,
                            'region_name': getattr( station_row, 'region_name', station_row.region ),
                            'station_name': getattr( station_row, 'station_name', station_row.station ),
                            'selection_reason': getattr( station_row, 'selection_reason', None ),
                            'latitude': getattr( station_row, 'latitude', None ),
                            'longitude': getattr( station_row, 'longitude', None ),
                            'cluster_label': getattr( station_row, 'cluster_label', None ),
                            'model': model_name,
                            'member': member_id,
                            'scenario': scenario_name,
                            'year': int( year ),
                            'variable': variable_name,
                            'target_driver': spec[ 'target_driver' ],
                            'native_units': spec[ 'native_units' ],
                            'target_units': spec[ 'target_units' ],
                            'conversion_note': spec[ 'conversion_note' ],
                            'relative_path': relative_path,
                            'file_url': f'{ base_url }/{ relative_path }',
                        } )

    manifest = pd.DataFrame( rows )
    return manifest


def build_nex_download_plan( 
    station_frame,
    models = None,
    model_variants = None,
    variables = None,
    variable_specs = None,
    scenario_years = None,
    base_url = NEX_GDDP_BASE_URL,
    data_prefix = NEX_GDDP_DATA_PREFIX,
):
    """ Build a download plan for NEX-GDDP CMIP6 assets based on the given station frame and parameters. """
    models = models or DEFAULT_NEX_MODELS
    model_variants = model_variants or DEFAULT_NEX_MODEL_VARIANTS
    variable_specs = variable_specs or DEFAULT_NEX_VARIABLE_SPECS
    variables = variables or list( variable_specs.keys( ) )
    scenario_years = scenario_years or DEFAULT_NEX_SCENARIO_YEARS

    bbox = build_station_bbox( station_frame )
    # note that the manifest will have one row per station-model-scenario-variable-year 
    # combination, so we summarize the manifest parameters here for a more concise download plan output
    plan = { 
        'base_url': base_url,
        'data_prefix': data_prefix,
        'models': [ { 'model': model_name, 'member': model_variants[ model_name ] } for model_name in models ],
        'variables': [ { 'variable': variable_name, **variable_specs[ variable_name ] } for variable_name in variables ],
        'scenario_years': [ { 'scenario': name, 'year_start': years[ 0 ], 'year_end': years[ 1 ] } for name, years in scenario_years.items( ) ],
        'station_count': int( station_frame[ [ 'region', 'station' ] ].drop_duplicates( ).shape[ 0 ] ),
        'bbox': bbox,
        'spatial_sampling_note': 'Prefer nearest valid coastal / land-adjacent 0.25-degree cell per station; use bbox only for Chesapeake-wide visual summary products.',
    }
    return plan


def build_unique_asset_table( manifest ):
    asset_cols = [ 
        'model',
        'member',
        'scenario',
        'year',
        'variable',
        'target_driver',
        'native_units',
        'target_units',
        'conversion_note',
        'relative_path',
        'file_url',
    ]
    asset_cols = [ col for col in asset_cols if col in manifest.columns ]

    unique_assets = manifest[ asset_cols ].drop_duplicates( ).sort_values( [ 'model', 'scenario', 'variable', 'year' ] ).reset_index( drop = True )
    return unique_assets


def save_manifest_outputs( manifest, station_frame, wobble_ranked, out_dir ):
    out_dir = Path( out_dir )
    out_dir.mkdir( parents = True, exist_ok = True )

    manifest_path = out_dir / 'nex_gddp_cmip6_station_asset_manifest.csv'
    unique_asset_path = out_dir / 'nex_gddp_cmip6_unique_assets.csv'
    station_path = out_dir / 'nex_gddp_cmip6_station_targets.csv'
    summary_path = out_dir / 'nex_gddp_cmip6_manifest_summary.csv'
    wobble_path = out_dir / 'nex_gddp_cmip6_wobble_candidates.csv'

    unique_assets = build_unique_asset_table( manifest )

    manifest.to_csv( manifest_path, index = False )
    unique_assets.to_csv( unique_asset_path, index = False )
    station_frame.to_csv( station_path, index = False )
    wobble_ranked.to_csv( wobble_path, index = False )

    summary = ( 
        unique_assets
        .groupby( [ 'model', 'scenario', 'variable' ], as_index = False )
        .agg( 
            n_files = ( 'file_url', 'size' ),
            year_start = ( 'year', 'min' ),
            year_end = ( 'year', 'max' ),
        )
    )
    summary.to_csv( summary_path, index = False )

    return { 
        'manifest_path': str( manifest_path.resolve( ) ),
        'unique_asset_path': str( unique_asset_path.resolve( ) ),
        'station_path': str( station_path.resolve( ) ),
        'summary_path': str( summary_path.resolve( ) ),
        'wobble_path': str( wobble_path.resolve( ) ),
        'n_manifest_rows': int( len( manifest ) ),
        'n_unique_assets': int( len( unique_assets ) ),
        'n_station_targets': int( station_frame[ [ 'region', 'station' ] ].drop_duplicates( ).shape[ 0 ] ),
    }
