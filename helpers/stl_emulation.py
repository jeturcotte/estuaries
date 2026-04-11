import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def build_monthly_station_history( daily_air_final, daily_water_final, station_keys, start_year = 1995 ):
    hist_air = daily_air_final.merge( station_keys, on = [ 'region', 'station' ], how = 'inner' ).copy( )
    hist_water = daily_water_final.merge( station_keys, on = [ 'region', 'station' ], how = 'inner' ).copy( )
    hist_air[ 'date' ] = pd.to_datetime( hist_air[ 'date' ], errors = 'coerce' )
    hist_water[ 'date' ] = pd.to_datetime( hist_water[ 'date' ], errors = 'coerce' )

    hist_air = hist_air.loc[ hist_air[ 'date' ].dt.year >= int( start_year ) ].copy( )
    hist_water = hist_water.loc[ hist_water[ 'date' ].dt.year >= int( start_year ) ].copy( )

    monthly_air = ( 
        hist_air
        .assign( month_date = lambda frame: frame[ 'date' ].dt.to_period( 'M' ).dt.to_timestamp( ) )
        .groupby( [ 'region', 'station', 'month_date' ], as_index = False )
        .agg( 
            air_temp = ( 'air_temp', 'mean' ),
            precip = ( 'precip', 'mean' ),
        )
    )
    monthly_water = ( 
        hist_water
        .assign( month_date = lambda frame: frame[ 'date' ].dt.to_period( 'M' ).dt.to_timestamp( ) )
        .groupby( [ 'region', 'station', 'month_date' ], as_index = False )
        .agg( 
            water_temp = ( 'water_temp', 'mean' ),
            salinity = ( 'salinity', 'mean' ),
            oxygen = ( 'oxygen', 'mean' ),
        )
    )

    monthly = monthly_air.merge( monthly_water, on = [ 'region', 'station', 'month_date' ], how = 'inner' )
    monthly = monthly.sort_values( [ 'region', 'station', 'month_date' ] ).reset_index( drop = True )
    monthly[ 'year' ] = monthly[ 'month_date' ].dt.year
    monthly[ 'month_num' ] = monthly[ 'month_date' ].dt.month
    return monthly


def stl_extend_series( series, forecast_end_year = 2100, seasonal_period = 12, trend_years = 10, min_months = 84 ):
    series = series.copy( ).sort_index( )
    series.index = pd.DatetimeIndex( series.index )
    series = series.asfreq( 'MS' )

    observed = pd.to_numeric( series, errors = 'coerce' )
    valid_count = int( observed.notna( ).sum( ) )
    if valid_count < int( min_months ):
        return None

    filled = observed.interpolate( method = 'time', limit_direction = 'both' )
    if filled.notna( ).sum( ) < int( min_months ):
        return None

    stl_fit = STL( filled, period = seasonal_period, robust = True ).fit( )
    trend = pd.Series( stl_fit.trend, index = filled.index )
    seasonal = pd.Series( stl_fit.seasonal, index = filled.index )
    fitted = trend + seasonal

    trend_tail = trend.dropna( ).tail( max( seasonal_period * int( trend_years ), seasonal_period * 3 ) )
    if len( trend_tail ) < seasonal_period * 2:
        return None

    x_tail = np.arange( len( trend_tail ), dtype = 'float64' )
    y_tail = np.asarray( trend_tail.to_numpy( ), dtype = 'float64' )
    slope, intercept = np.polyfit( x_tail, y_tail, deg = 1 )

    seasonal_template = seasonal.groupby( seasonal.index.month ).mean( )

    last_month = filled.index.max( )
    future_end = pd.Timestamp( int( forecast_end_year ), 12, 1 )
    if future_end <= last_month:
        future_index = pd.DatetimeIndex( [ ] )

    else:
        future_index = pd.date_range( last_month + pd.offsets.MonthBegin( 1 ), future_end, freq = 'MS' )

    if len( future_index ) > 0:
        x_future = np.arange( len( trend_tail ), len( trend_tail ) + len( future_index ), dtype = 'float64' )
        trend_future = intercept + slope * x_future
        seasonal_future = np.asarray( [ seasonal_template.get( month.month, 0.0 ) for month in future_index ], dtype = 'float64' )
        forecast = trend_future + seasonal_future

    else:
        trend_future = np.asarray( [ ], dtype = 'float64' )
        seasonal_future = np.asarray( [ ], dtype = 'float64' )
        forecast = np.asarray( [ ], dtype = 'float64' )

    history_frame = pd.DataFrame( { 
        'month_date': filled.index,
        'observed': observed.to_numpy( ),
        'filled': filled.to_numpy( ),
        'trend': trend.to_numpy( ),
        'seasonal': seasonal.to_numpy( ),
        'fitted': fitted.to_numpy( ),
        'is_future': False,
    } )
    future_frame = pd.DataFrame( { 
        'month_date': future_index,
        'observed': np.nan,
        'filled': np.nan,
        'trend': trend_future,
        'seasonal': seasonal_future,
        'fitted': forecast,
        'is_future': True,
    } )

    out = pd.concat( [ history_frame, future_frame ], ignore_index = True )
    out[ 'year' ] = out[ 'month_date' ].dt.year
    out[ 'month_num' ] = out[ 'month_date' ].dt.month
    out[ 'trend_slope_per_month' ] = float( slope )
    out[ 'trend_slope_per_year' ] = float( slope * 12.0 )
    return out


def run_stl_emulation( 
    monthly_history,
    station_meta,
    station_keys,
    forecast_end_year = 2100,
    trend_years = 10,
    min_months = 84,
):
    variable_map = { 
        'air_temp': 'Air Temp',
        'precip': 'Precip',
        'water_temp': 'Water Temp',
        'salinity': 'Salinity',
        'oxygen': 'Dissolved Oxygen',
    }

    monthly_history = monthly_history.merge( station_meta, on = [ 'region', 'station' ], how = 'left' )
    station_keys = station_keys.drop_duplicates( ).reset_index( drop = True )

    projection_parts = [ ]
    summary_rows = [ ]

    for station_key in station_keys.itertuples( index = False ):
        station_monthly = monthly_history.loc[ 
            ( monthly_history[ 'region' ] == station_key.region )
            & ( monthly_history[ 'station' ] == station_key.station )
        ].copy( )
        if len( station_monthly ) == 0:
            continue

        station_info = station_monthly.iloc[ 0 ]
        for variable, variable_label in variable_map.items( ):
            series = station_monthly.set_index( 'month_date' )[ variable ]
            projected = stl_extend_series( 
                series,
                forecast_end_year = forecast_end_year,
                trend_years = trend_years,
                min_months = min_months,
            )
            if projected is None:
                continue

            projected[ 'region' ] = station_key.region
            projected[ 'station' ] = station_key.station
            projected[ 'station_name' ] = station_info.get( 'station_name', station_key.station )
            projected[ 'region_name' ] = station_info.get( 'region_name', station_key.region )
            projected[ 'cluster_label' ] = station_info.get( 'cluster_label', np.nan )
            projected[ 'variable' ] = variable
            projected[ 'variable_label' ] = variable_label
            projection_parts.append( projected )

            hist_tail = projected.loc[ ~projected[ 'is_future' ] & projected[ 'observed' ].notna( ) & ( projected[ 'year' ] >= projected[ 'year' ].max( ) - 4 ) ]
            future_2090s = projected.loc[ projected[ 'is_future' ] & ( projected[ 'year' ] >= 2090 ) ]
            summary_rows.append( { 
                'region': station_key.region,
                'station': station_key.station,
                'station_name': station_info.get( 'station_name', station_key.station ),
                'region_name': station_info.get( 'region_name', station_key.region ),
                'cluster_label': station_info.get( 'cluster_label', np.nan ),
                'variable': variable,
                'variable_label': variable_label,
                'recent_mean_5y': float( hist_tail[ 'observed' ].mean( ) ) if len( hist_tail ) > 0 else np.nan,
                'forecast_mean_2090s': float( future_2090s[ 'fitted' ].mean( ) ) if len( future_2090s ) > 0 else np.nan,
                'forecast_delta_2090s': float( future_2090s[ 'fitted' ].mean( ) - hist_tail[ 'observed' ].mean( ) ) if len( hist_tail ) > 0 and len( future_2090s ) > 0 else np.nan,
                'trend_slope_per_year': float( projected[ 'trend_slope_per_year' ].iloc[ 0 ] ),
            } )

    projection_long = pd.concat( projection_parts, ignore_index = True ) if len( projection_parts ) > 0 else pd.DataFrame( )
    summary = pd.DataFrame( summary_rows )
    return { 
        'monthly_history': monthly_history,
        'projection_long': projection_long,
        'summary': summary,
        'station_keys': station_keys,
    }
