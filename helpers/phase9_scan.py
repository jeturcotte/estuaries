from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _share_below( values, threshold ):
    values = pd.to_numeric( pd.Series( values ), errors = 'coerce' ).dropna( )
    if len( values ) == 0:
        return np.nan

    return float( ( values < threshold ).mean( ) )


def run_phase9_station_scan( 
    station_baseline_display,
    daily_air_final,
    daily_water_final,
    first_event,
    flagged_station_keys_final,
    phase9_scenario_path_csv,
    build_phase9_demo_paths,
    add_phase9_driver_rolls,
    build_phase9_station_year_features,
    phase9_classify_regime_row,
    phase9_rolling_window_years,
    phase6_context_table,
    phase6_selected_features,
    phase6_selected_fill_values,
    phase6_selected_model,
    phase7_targets,
    phase7_feature_store,
    phase7_fill_store,
    phase7_model_store,
    station_baseline = None,
    properties_baseline = None,
    phase9_scenario_paths = None,
    phase9_scenarios = None,
    phase9_future_year_min = None,
    phase9_future_year_max = None,
    phase9_future_dates = None,
    phase9_properties_baseline = None,
    phase9_cluster_lookup = None,
    region_codes = None,
    top_n = 12,
    plot_n = 6,
    hypoxia_threshold = 2.0,
    material_share = 0.05,
    progress_every = 20,
):
    if phase9_scenario_paths is None:
        if Path( phase9_scenario_path_csv ).exists( ):
            phase9_scenario_paths = pd.read_csv( phase9_scenario_path_csv )
            path_source = f'external_csv: { phase9_scenario_path_csv }'

        else:
            phase9_scenario_paths = build_phase9_demo_paths( start_year = 2026, end_year = 2100 )
            path_source = 'demo_ramps'

        phase9_scenario_paths[ 'scenario' ] = phase9_scenario_paths[ 'scenario' ].astype( str )
        phase9_scenario_paths[ 'year' ] = pd.to_numeric( phase9_scenario_paths[ 'year' ], errors = 'coerce' ).astype( 'Int64' )
        for col, fill_value in [ 
            ( 'air_temp_add_c', 0.0 ),
            ( 'precip_mult', 1.0 ),
            ( 'wind_speed_mult', 1.0 ),
            ( 'solar_mult', 1.0 ),
        ]:
            if col not in phase9_scenario_paths.columns:
                phase9_scenario_paths[ col ] = fill_value

            phase9_scenario_paths[ col ] = pd.to_numeric( phase9_scenario_paths[ col ], errors = 'coerce' ).fillna( fill_value )

    else:
        phase9_scenario_paths = phase9_scenario_paths.copy( )
        path_source = 'existing_phase9_scenario_paths'

    if phase9_scenarios is None:
        phase9_scenarios = sorted( phase9_scenario_paths[ 'scenario' ].dropna( ).unique( ).tolist( ) )

    if phase9_future_year_min is None:
        phase9_future_year_min = int( phase9_scenario_paths[ 'year' ].dropna( ).min( ) )

    if phase9_future_year_max is None:
        phase9_future_year_max = int( phase9_scenario_paths[ 'year' ].dropna( ).max( ) )

    if phase9_future_dates is None:
        phase9_future_dates = pd.DataFrame( { 
            'date': pd.date_range( f'{ phase9_future_year_min }-01-01', f'{ phase9_future_year_max }-12-31', freq = 'D' ),
        } )
        phase9_future_dates[ 'year' ] = phase9_future_dates[ 'date' ].dt.year
        phase9_future_dates[ 'doy' ] = phase9_future_dates[ 'date' ].dt.dayofyear.clip( upper = 365 )
        phase9_future_dates = phase9_future_dates.loc[ phase9_future_dates[ 'year' ].isin( phase9_scenario_paths[ 'year' ].dropna( ).astype( int ) ) ].copy( )

    if phase9_properties_baseline is None:
        if properties_baseline is not None:
            phase9_properties_baseline = properties_baseline.copy( )

        else:
            phase9_properties_baseline = ( 
                daily_water_final
                .groupby( [ 'region', 'station' ], as_index = False )
                .agg( 
                    water_temp_baseline = ( 'water_temp_baseline', 'mean' ),
                    salinity_baseline = ( 'salinity_baseline', 'mean' ),
                    oxygen_baseline = ( 'oxygen_baseline', 'mean' ),
                    ph_baseline = ( 'ph_baseline', 'mean' ),
                    depth_baseline = ( 'depth_baseline', 'mean' ),
                )
            )

    if phase9_cluster_lookup is None:
        if station_baseline is None:
            raise ValueError( 'station_baseline is required when phase9_cluster_lookup is not provided.' )

        phase9_cluster_lookup = station_baseline[ [ 'region', 'station', 'cluster' ] ].copy( )
        phase9_cluster_lookup[ 'cluster_code' ] = pd.to_numeric( phase9_cluster_lookup[ 'cluster' ], errors = 'coerce' )
        phase9_cluster_lookup = phase9_cluster_lookup.drop( columns = [ 'cluster' ] )

    station_meta = ( 
        station_baseline_display[ [ 'region', 'station', 'region_name', 'station_name', 'cluster', 'cluster_label', 'cluster_name' ] ]
        .drop_duplicates( )
        .copy( )
    )
    available_keys = ( 
        daily_air_final[ [ 'region', 'station' ] ]
        .drop_duplicates( )
        .merge( daily_water_final[ [ 'region', 'station' ] ].drop_duplicates( ), on = [ 'region', 'station' ], how = 'inner' )
    )
    station_meta = available_keys.merge( station_meta, on = [ 'region', 'station' ], how = 'left' )
    station_meta = station_meta.merge( first_event[ [ 'region', 'station', 'event_date' ] ], on = [ 'region', 'station' ], how = 'left' )
    station_meta = station_meta.merge( flagged_station_keys_final.assign( is_holdout = True ), on = [ 'region', 'station' ], how = 'left' )
    station_meta[ 'is_holdout' ] = station_meta[ 'is_holdout' ].fillna( False )
    station_meta[ 'is_train' ] = ~station_meta[ 'is_holdout' ]

    if region_codes is not None and len( region_codes ) > 0:
        region_codes_norm = { str( code ).strip( ).lower( ) for code in region_codes }
        station_meta = station_meta.loc[ station_meta[ 'region' ].astype( str ).str.lower( ).isin( region_codes_norm ) ].copy( )

    else:
        region_codes_norm = None

    station_keys = station_meta[ [ 'region', 'station' ] ].drop_duplicates( ).reset_index( drop = True )

    hist_air = daily_air_final.merge( station_keys, on = [ 'region', 'station' ], how = 'inner' ).copy( )
    hist_water = daily_water_final.merge( station_keys, on = [ 'region', 'station' ], how = 'inner' ).copy( )
    hist_air[ 'date' ] = pd.to_datetime( hist_air[ 'date' ], errors = 'coerce' )
    hist_water[ 'date' ] = pd.to_datetime( hist_water[ 'date' ], errors = 'coerce' )

    hist_join = hist_air.merge( 
        hist_water[ [ 'region', 'station', 'date', 'water_temp', 'salinity', 'oxygen', 'depth' ] ],
        on = [ 'region', 'station', 'date' ],
        how = 'inner',
    )
    hist_join[ 'year' ] = hist_join[ 'date' ].dt.year
    hist_join[ 'month_num' ] = hist_join[ 'date' ].dt.month
    hist_join[ 'is_warm_season' ] = hist_join[ 'month_num' ].between( 6, 9 )

    history_annual = ( 
        hist_join
        .groupby( [ 'region', 'station', 'year' ], as_index = False )
        .agg( 
            air_temp_mean = ( 'air_temp', 'mean' ),
            precip_mean = ( 'precip', 'mean' ),
            water_temp_abs_annual_mean = ( 'water_temp', 'mean' ),
            salinity_abs_annual_mean = ( 'salinity', 'mean' ),
            oxygen_abs_annual_mean = ( 'oxygen', 'mean' ),
            depth_abs_annual_mean = ( 'depth', 'mean' ),
        )
    )
    history_warm = ( 
        hist_join.loc[ hist_join[ 'is_warm_season' ] ]
        .groupby( [ 'region', 'station', 'year' ], as_index = False )
        .agg( 
            oxygen_warm_min_abs = ( 'oxygen', 'min' ),
            oxygen_warm_hypoxic_day_share_abs = ( 'oxygen', lambda values: _share_below( values, hypoxia_threshold ) ),
        )
    )
    history_annual = history_annual.merge( history_warm, on = [ 'region', 'station', 'year' ], how = 'left' )
    history_annual = history_annual.merge( phase9_properties_baseline, on = [ 'region', 'station' ], how = 'left' )
    history_annual = history_annual.merge( station_meta, on = [ 'region', 'station' ], how = 'left' )
    history_annual[ 'scenario' ] = 'observed_history'
    history_annual[ 'oxygen_plot_abs' ] = history_annual[ 'oxygen_warm_min_abs' ]
    history_annual.loc[ history_annual[ 'oxygen_plot_abs' ].isna( ), 'oxygen_plot_abs' ] = history_annual.loc[ history_annual[ 'oxygen_plot_abs' ].isna( ), 'oxygen_abs_annual_mean' ]

    future_template = ( 
        pd.DataFrame( { 'scenario': phase9_scenarios, '_tmp': 1 } )
        .merge( phase9_future_dates.assign( _tmp = 1 ), on = '_tmp', how = 'inner' )
        .drop( columns = [ '_tmp' ] )
    )

    future_parts = [ ]
    station_total = len( station_keys )

    for idx, station_key in enumerate( station_keys.itertuples( index = False ), start = 1 ):
        if idx == 1 or idx == station_total or idx % progress_every == 0:
            print( f'phase9 scan progress: { idx } / { station_total } | { station_key.region } / { station_key.station }' )

        station_air = hist_air.loc[ 
            ( hist_air[ 'region' ] == station_key.region )
            & ( hist_air[ 'station' ] == station_key.station )
        ].copy( )
        if len( station_air ) == 0:
            continue

        station_air[ 'doy' ] = station_air[ 'date' ].dt.dayofyear.clip( upper = 365 )
        driver_climatology = ( 
            station_air
            .groupby( [ 'region', 'station', 'doy' ], as_index = False )
            .agg( 
                air_temp = ( 'air_temp', 'mean' ),
                precip = ( 'precip', 'mean' ),
                wind_speed = ( 'wind_speed', 'mean' ),
                solar = ( 'solar', 'mean' ),
            )
        )
        driver_mean = ( 
            station_air
            .groupby( [ 'region', 'station' ], as_index = False )
            .agg( 
                air_temp_station_mean = ( 'air_temp', 'mean' ),
                precip_station_mean = ( 'precip', 'mean' ),
                wind_speed_station_mean = ( 'wind_speed', 'mean' ),
                solar_station_mean = ( 'solar', 'mean' ),
            )
        )

        future_daily = future_template.copy( )
        future_daily[ 'region' ] = station_key.region
        future_daily[ 'station' ] = station_key.station
        future_daily = future_daily.merge( driver_climatology, on = [ 'region', 'station', 'doy' ], how = 'left' )
        future_daily = future_daily.merge( driver_mean, on = [ 'region', 'station' ], how = 'left' )
        future_daily = future_daily.merge( phase9_scenario_paths, on = [ 'scenario', 'year' ], how = 'left' )

        for col, station_fill_col in [ 
            ( 'air_temp', 'air_temp_station_mean' ),
            ( 'precip', 'precip_station_mean' ),
            ( 'wind_speed', 'wind_speed_station_mean' ),
            ( 'solar', 'solar_station_mean' ),
        ]:
            future_daily[ col ] = future_daily[ col ].fillna( future_daily[ station_fill_col ] )

        future_daily[ 'air_temp' ] = future_daily[ 'air_temp' ] + future_daily[ 'air_temp_add_c' ]
        future_daily[ 'precip' ] = future_daily[ 'precip' ] * future_daily[ 'precip_mult' ]
        future_daily[ 'wind_speed' ] = future_daily[ 'wind_speed' ] * future_daily[ 'wind_speed_mult' ]
        future_daily[ 'solar' ] = future_daily[ 'solar' ] * future_daily[ 'solar_mult' ]
        future_daily = add_phase9_driver_rolls( future_daily )

        future_daily = future_daily.merge( phase9_properties_baseline, on = [ 'region', 'station' ], how = 'left' )
        future_daily = future_daily.merge( phase6_context_table, on = [ 'region', 'station' ], how = 'left' )
        future_daily = future_daily.merge( phase9_cluster_lookup, on = [ 'region', 'station' ], how = 'left' )

        future_daily[ 'air_temp_minus_water_temp_baseline' ] = future_daily[ 'air_temp' ] - future_daily[ 'water_temp_baseline' ]
        future_daily[ 'air_temp_r7d_minus_water_temp_baseline' ] = future_daily[ 'air_temp_r7d' ] - future_daily[ 'water_temp_baseline' ]
        future_daily[ 'air_temp_r28d_minus_water_temp_baseline' ] = future_daily[ 'air_temp_r28d' ] - future_daily[ 'water_temp_baseline' ]
        future_daily[ 'air_temp_r1d_minus_air_temp_r28d' ] = future_daily[ 'air_temp_r1d' ] - future_daily[ 'air_temp_r28d' ]
        future_daily[ 'air_temp_r7d_minus_air_temp_r28d' ] = future_daily[ 'air_temp_r7d' ] - future_daily[ 'air_temp_r28d' ]
        future_daily[ 'wind_speed_x_air_temp' ] = future_daily[ 'wind_speed' ] * future_daily[ 'air_temp' ]
        future_daily[ 'solar_x_air_temp' ] = future_daily[ 'solar' ] * future_daily[ 'air_temp' ]

        for col in phase6_selected_features:
            if col not in future_daily.columns:
                future_daily[ col ] = np.nan

        X_p6 = future_daily[ phase6_selected_features ].copy( ).fillna( phase6_selected_fill_values.reindex( phase6_selected_features ) )
        future_daily[ 'delta_water_temp_pred_p6' ] = phase6_selected_model.predict( X_p6 )
        future_daily[ 'water_temp_pred' ] = future_daily[ 'water_temp_baseline' ] + future_daily[ 'delta_water_temp_pred_p6' ]

        station_future_year = build_phase9_station_year_features( future_daily )
        for target in phase7_targets:
            model_t = phase7_model_store.get( target )
            feature_cols_t = phase7_feature_store.get( target, [ ] )
            fill_values_t = phase7_fill_store.get( target, pd.Series( dtype = 'float64' ) )

            if model_t is None or len( feature_cols_t ) == 0:
                continue

            X_target_t = station_future_year[ feature_cols_t ].copy( ).fillna( fill_values_t.reindex( feature_cols_t ) )
            station_future_year[ f'{ target }_pred' ] = model_t.predict( X_target_t )

        station_future_year[ 'water_temp_abs_annual_mean' ] = station_future_year[ 'water_temp_baseline_annual_mean' ] + station_future_year[ 'delta_water_temp_pred_p6_mean' ]
        station_future_year[ 'salinity_abs_annual_mean' ] = station_future_year[ 'salinity_baseline_annual_mean' ] + station_future_year.get( 'delta_salinity_pred', 0.0 )
        if 'delta_oxygen_pred' in station_future_year.columns:
            station_future_year[ 'oxygen_abs_annual_mean' ] = station_future_year[ 'oxygen_baseline_annual_mean' ] + station_future_year[ 'delta_oxygen_pred' ]

        else:
            station_future_year[ 'oxygen_abs_annual_mean' ] = np.nan

        station_future_year[ 'oxygen_warm_min_abs' ] = station_future_year.get( 'oxygen_warm_min_pred', np.nan )
        station_future_year[ 'oxygen_warm_hypoxic_day_share_abs' ] = station_future_year.get( 'oxygen_warm_hypoxic_day_share_pred', np.nan )
        station_future_year[ 'oxygen_plot_abs' ] = station_future_year[ 'oxygen_warm_min_abs' ]
        station_future_year.loc[ station_future_year[ 'oxygen_plot_abs' ].isna( ), 'oxygen_plot_abs' ] = station_future_year.loc[ station_future_year[ 'oxygen_plot_abs' ].isna( ), 'oxygen_abs_annual_mean' ]
        station_future_year = station_future_year.merge( station_meta, on = [ 'region', 'station' ], how = 'left' )
        future_parts.append( station_future_year )

    future_station_year = pd.concat( future_parts, ignore_index = True ) if len( future_parts ) > 0 else pd.DataFrame( )

    metric_roll_map = { 
        'air_temp_mean': 'air_temp_roll5y',
        'precip_mean': 'precip_roll5y',
        'water_temp_abs_annual_mean': 'water_temp_roll5y',
        'salinity_abs_annual_mean': 'salinity_roll5y',
        'oxygen_abs_annual_mean': 'oxygen_roll5y',
        'oxygen_plot_abs': 'oxygen_plot_roll5y',
        'oxygen_warm_hypoxic_day_share_abs': 'oxygen_hypoxic_share_roll5y',
    }

    history_plot = history_annual.copy( )
    for source_col, out_col in metric_roll_map.items( ):
        history_plot[ out_col ] = ( 
            history_plot
            .sort_values( [ 'region', 'station', 'year' ] )
            .groupby( [ 'region', 'station' ] )[ source_col ]
            .transform( lambda values: values.rolling( window = phase9_rolling_window_years, min_periods = 3 ).mean( ) )
        )

    history_plot[ 'mean_annual_water_temp' ] = history_plot[ 'water_temp_roll5y' ]
    history_plot[ 'mean_annual_salinity' ] = history_plot[ 'salinity_roll5y' ]
    history_plot[ 'mean_annual_oxygen' ] = history_plot[ 'oxygen_roll5y' ]
    history_plot[ 'mean_annual_depth' ] = history_plot[ 'depth_baseline' ]
    history_plot = pd.concat( [ history_plot, history_plot.apply( phase9_classify_regime_row, axis = 1 ) ], axis = 1 )

    projection_paths = [ ]
    for scenario_name in phase9_scenarios:
        hist_path = history_annual.copy( )
        hist_path[ 'scenario' ] = scenario_name
        future_path = future_station_year.loc[ future_station_year[ 'scenario' ] == scenario_name ].copy( )
        path_frame = pd.concat( [ hist_path, future_path ], ignore_index = True, sort = False )
        path_frame = path_frame.sort_values( [ 'region', 'station', 'year' ] ).reset_index( drop = True )

        for source_col, out_col in metric_roll_map.items( ):
            path_frame[ out_col ] = ( 
                path_frame
                .groupby( [ 'region', 'station' ] )[ source_col ]
                .transform( lambda values: values.rolling( window = phase9_rolling_window_years, min_periods = 3 ).mean( ) )
            )

        path_frame[ 'mean_annual_water_temp' ] = path_frame[ 'water_temp_roll5y' ]
        path_frame[ 'mean_annual_salinity' ] = path_frame[ 'salinity_roll5y' ]
        path_frame[ 'mean_annual_oxygen' ] = path_frame[ 'oxygen_roll5y' ]
        path_frame[ 'mean_annual_depth' ] = path_frame[ 'depth_baseline' ]
        path_frame = pd.concat( [ path_frame, path_frame.apply( phase9_classify_regime_row, axis = 1 ) ], axis = 1 )
        projection_paths.append( path_frame )

    projection_plot = pd.concat( projection_paths, ignore_index = True ) if len( projection_paths ) > 0 else pd.DataFrame( )
    projection_future_only = projection_plot.loc[ projection_plot[ 'year' ] >= phase9_future_year_min ].copy( )

    crossing_summary = ( 
        projection_future_only
        .loc[ 
            projection_future_only[ 'regime_roll5y' ].notna( )
            & projection_future_only[ 'cluster' ].notna( )
            & ( projection_future_only[ 'regime_roll5y' ].astype( 'Int64' ) != projection_future_only[ 'cluster' ].astype( 'Int64' ) )
        ]
        .groupby( [ 'region', 'station', 'scenario' ], as_index = False )[ 'year' ]
        .min( )
        .rename( columns = { 'year': 'first_regime_cross_year' } )
    )
    crossing_candidates = station_meta.merge( crossing_summary, on = [ 'region', 'station' ], how = 'inner' )
    crossing_candidates[ 'first_regime_cross_decade' ] = ( crossing_candidates[ 'first_regime_cross_year' ] // 10 ) * 10
    crossing_candidates = crossing_candidates.sort_values( [ 'scenario', 'first_regime_cross_year', 'region', 'station' ] ).reset_index( drop = True )

    history_hypoxia = ( 
        history_annual
        .groupby( [ 'region', 'station' ], as_index = False )
        .agg( 
            hist_hypoxic_share_peak = ( 'oxygen_warm_hypoxic_day_share_abs', 'max' ),
            hist_hypoxic_share_mean = ( 'oxygen_warm_hypoxic_day_share_abs', 'mean' ),
            hist_warm_min_floor = ( 'oxygen_warm_min_abs', 'min' ),
            hist_warm_min_mean = ( 'oxygen_warm_min_abs', 'mean' ),
        )
    )
    history_hypoxia[ 'hist_hypoxia_any' ] = ( 
        history_hypoxia[ 'hist_hypoxic_share_peak' ].fillna( 0.0 ) > 0.0
    ) | ( 
        history_hypoxia[ 'hist_warm_min_floor' ].fillna( np.inf ) < hypoxia_threshold
    )

    future_hypoxia = ( 
        projection_future_only
        .groupby( [ 'region', 'station', 'scenario' ], as_index = False )
        .agg( 
            future_hypoxic_share_peak = ( 'oxygen_warm_hypoxic_day_share_abs', 'max' ),
            future_hypoxic_share_mean = ( 'oxygen_warm_hypoxic_day_share_abs', 'mean' ),
            future_warm_min_floor = ( 'oxygen_warm_min_abs', 'min' ),
            future_warm_min_mean = ( 'oxygen_warm_min_abs', 'mean' ),
        )
    )
    future_hypoxia_2090s = ( 
        projection_future_only.loc[ projection_future_only[ 'year' ] >= 2090 ]
        .groupby( [ 'region', 'station', 'scenario' ], as_index = False )
        .agg( 
            future_hypoxic_share_2090s = ( 'oxygen_warm_hypoxic_day_share_abs', 'mean' ),
            future_warm_min_mean_2090s = ( 'oxygen_warm_min_abs', 'mean' ),
        )
    )
    hypoxia_candidates = station_meta.merge( future_hypoxia, on = [ 'region', 'station' ], how = 'inner' )
    hypoxia_candidates = hypoxia_candidates.merge( future_hypoxia_2090s, on = [ 'region', 'station', 'scenario' ], how = 'left' )
    hypoxia_candidates = hypoxia_candidates.merge( history_hypoxia, on = [ 'region', 'station' ], how = 'left' )
    hypoxia_candidates[ 'new_hypoxia_flag' ] = ( 
        ~hypoxia_candidates[ 'hist_hypoxia_any' ].fillna( False )
    ) & ( 
        ( hypoxia_candidates[ 'future_hypoxic_share_peak' ].fillna( 0.0 ) >= material_share )
        | ( hypoxia_candidates[ 'future_warm_min_floor' ].fillna( np.inf ) < hypoxia_threshold )
    )
    hypoxia_candidates[ 'hypoxia_share_increase_peak' ] = hypoxia_candidates[ 'future_hypoxic_share_peak' ].fillna( 0.0 ) - hypoxia_candidates[ 'hist_hypoxic_share_peak' ].fillna( 0.0 )
    hypoxia_candidates[ 'hypoxia_share_increase_2090s' ] = hypoxia_candidates[ 'future_hypoxic_share_2090s' ].fillna( 0.0 ) - hypoxia_candidates[ 'hist_hypoxic_share_mean' ].fillna( 0.0 )
    hypoxia_candidates[ 'warm_min_drop_mean_2090s' ] = hypoxia_candidates[ 'hist_warm_min_mean' ] - hypoxia_candidates[ 'future_warm_min_mean_2090s' ]
    hypoxia_candidates[ 'hypoxia_intensifies_flag' ] = ( 
        hypoxia_candidates[ 'hypoxia_share_increase_peak' ].fillna( 0.0 ) >= material_share
    ) | ( 
        hypoxia_candidates[ 'warm_min_drop_mean_2090s' ].fillna( 0.0 ) >= 0.5
    )
    hypoxia_candidates = hypoxia_candidates.sort_values( 
        [ 'scenario', 'new_hypoxia_flag', 'hypoxia_share_increase_peak', 'future_hypoxic_share_peak', 'future_warm_min_floor' ],
        ascending = [ True, False, False, False, True ],
    ).reset_index( drop = True )

    crossing_display = crossing_candidates[ [ 
        'scenario',
        'region',
        'station',
        'station_name',
        'region_name',
        'cluster_label',
        'is_holdout',
        'first_regime_cross_year',
        'first_regime_cross_decade',
    ] ].head( top_n )
    hypoxia_display = hypoxia_candidates[ [ 
        'scenario',
        'region',
        'station',
        'station_name',
        'region_name',
        'cluster_label',
        'is_holdout',
        'new_hypoxia_flag',
        'hypoxia_intensifies_flag',
        'hist_hypoxic_share_peak',
        'future_hypoxic_share_peak',
        'hypoxia_share_increase_peak',
        'hist_warm_min_floor',
        'future_warm_min_floor',
    ] ].head( top_n )

    plot_station_keys = pd.concat( [ 
        crossing_display[ [ 'region', 'station' ] ],
        hypoxia_display[ [ 'region', 'station' ] ],
    ], ignore_index = True ).drop_duplicates( ).head( plot_n )
    plot_studies = station_meta.merge( plot_station_keys, on = [ 'region', 'station' ], how = 'inner' )

    return { 
        'path_source': path_source,
        'region_codes_norm': region_codes_norm,
        'station_meta': station_meta,
        'station_keys': station_keys,
        'scenarios': phase9_scenarios,
        'future_year_min': phase9_future_year_min,
        'future_year_max': phase9_future_year_max,
        'future_station_year': future_station_year,
        'history_annual': history_annual,
        'history_plot': history_plot,
        'projection_plot': projection_plot,
        'projection_future_only': projection_future_only,
        'crossing_candidates': crossing_candidates,
        'hypoxia_candidates': hypoxia_candidates,
        'crossing_display': crossing_display,
        'hypoxia_display': hypoxia_display,
        'plot_studies': plot_studies,
    }


def plot_phase9_scan_portraits( scan_result, station_baseline_display ):
    plot_studies = scan_result[ 'plot_studies' ]
    history_plot = scan_result[ 'history_plot' ]
    projection_plot = scan_result[ 'projection_plot' ]
    scenarios = scan_result[ 'scenarios' ]
    future_year_min = scan_result[ 'future_year_min' ]

    scenario_palette = dict( zip( scenarios, sns.color_palette( 'Set2', n_colors = len( scenarios ) ) ) )
    cluster_label_map = ( 
        station_baseline_display[ [ 'cluster', 'cluster_label' ] ]
        .dropna( subset = [ 'cluster' ] )
        .drop_duplicates( )
        .assign( cluster = lambda frame: frame[ 'cluster' ].astype( int ) )
        .set_index( 'cluster' )[ 'cluster_label' ]
        .to_dict( )
    )

    for station_row in plot_studies.itertuples( index = False ):
        hist_case = history_plot.loc[ 
            ( history_plot[ 'region' ] == station_row.region )
            & ( history_plot[ 'station' ] == station_row.station )
        ].copy( )
        scen_case = projection_plot.loc[ 
            ( projection_plot[ 'region' ] == station_row.region )
            & ( projection_plot[ 'station' ] == station_row.station )
        ].copy( )

        if len( hist_case ) == 0 or len( scen_case ) == 0:
            continue

        flip_label = 'no observed transition'
        if pd.notna( station_row.event_date ):
            flip_label = f'first observed flip { pd.to_datetime( station_row.event_date ).date( ) }'

        fig, axes = plt.subplots( 7, 1, figsize = ( 14, 18 ), sharex = True )
        fig.suptitle( 
            f"Phase 9 Scan Portrait | { station_row.station_name } ({ station_row.region_name }) | baseline { station_row.cluster_label } | { flip_label }",
            fontsize = 14,
            y = 0.995,
        )

        axes[ 0 ].plot( 
            hist_case[ 'year' ],
            hist_case[ 'regime_roll5y' ],
            color = 'black',
            linewidth = 2.2,
            label = 'observed 5y regime',
        )
        for scenario_name in scenarios:
            sub = scen_case.loc[ 
                ( scen_case[ 'scenario' ] == scenario_name )
                & ( scen_case[ 'year' ] >= future_year_min )
            ].copy( )
            axes[ 0 ].plot( 
                sub[ 'year' ],
                sub[ 'regime_roll5y' ],
                color = scenario_palette[ scenario_name ],
                linewidth = 1.8,
                alpha = 0.95,
                label = scenario_name,
            )

        cluster_ticks = sorted( [ int( val ) for val in pd.Series( pd.concat( [ hist_case[ 'regime_roll5y' ], scen_case[ 'regime_roll5y' ] ] ) ).dropna( ).astype( int ).unique( ) ] )
        if len( cluster_ticks ) > 0:
            axes[ 0 ].set_yticks( cluster_ticks )
            axes[ 0 ].set_yticklabels( [ cluster_label_map.get( tick, f'C{ tick }' ) for tick in cluster_ticks ] )

        if pd.notna( station_row.cluster ):
            axes[ 0 ].axhline( float( station_row.cluster ), color = '#555555', linestyle = '--', linewidth = 1.0 )

        axes[ 0 ].set_ylabel( 'Regime' )
        axes[ 0 ].legend( loc = 'upper left', ncol = 3 )

        plot_specs = [ 
            ( 'air_temp_roll5y', 'Air Temp ( 5y mean )' ),
            ( 'precip_roll5y', 'Precip ( 5y mean )' ),
            ( 'water_temp_roll5y', 'Water Temp ( 5y mean )' ),
            ( 'salinity_roll5y', 'Salinity ( 5y mean )' ),
            ( 'oxygen_plot_roll5y', 'Oxygen Signature ( 5y mean )' ),
            ( 'oxygen_hypoxic_share_roll5y', 'Warm Hypoxic Share ( 5y mean )' ),
        ]

        for ax, ( col, ylabel ) in zip( axes[ 1: ], plot_specs ):
            ax.plot( hist_case[ 'year' ], hist_case[ col ], color = 'black', linewidth = 2.2, label = 'observed' )

            for scenario_name in scenarios:
                sub = scen_case.loc[ 
                    ( scen_case[ 'scenario' ] == scenario_name )
                    & ( scen_case[ 'year' ] >= future_year_min )
                ].copy( )
                ax.plot( 
                    sub[ 'year' ],
                    sub[ col ],
                    color = scenario_palette[ scenario_name ],
                    linewidth = 1.8,
                    alpha = 0.95,
                    label = scenario_name,
                )

            ax.axvline( future_year_min, color = '#666666', linestyle = '--', linewidth = 0.9 )
            ax.set_ylabel( ylabel )

        axes[ -1 ].set_xlabel( 'Year' )
        plt.tight_layout( )
        plt.show( )
