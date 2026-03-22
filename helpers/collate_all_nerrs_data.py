#!/usr/bin/env python3
"""
Rebuild hourly collated NERRS data from station files (pre-ERA5 merge).

Pipeline:
1) Discover and classify station files from filename patterns.
2) Keep only stations with both WQ and NUT file presence.
3) Ingest WQ/NUT/MET into SQLite normalized tables.
4) Build hourly collated grid from WQ minute==00 rows.
5) Expand nutrients via backward as-of (7 days).
6) Enrich MET via backward as-of (1 hour) with station -> region -> cross-region fallback.
7) Write final + intermediate CSV outputs.
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path( __file__ ).resolve( ).parents[ 1 ]
# default source folder for raw nerrs station files
DEFAULT_DATA_DIR = REPO_ROOT / 'Data' / 'NERRS' / 'nerrs_most_stations'
# default target folder for collated nerrs outputs
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'Data' / 'NERRS'
# transient sqlite file for merge work
DEFAULT_SQLITE_PATH = DEFAULT_OUTPUT_DIR / 'nerrs_1h_collate.merge.sqlite'

# final collated hourly nerrs table
FINAL_OUTPUT_NAME = 'nerrs.collated.1h.csv'
# direct nutrient observations before asof fill
NUTRIENT_OBSERVED_NAME = 'nerrs.nutrients_observed.clean.csv'
# nutrient table after seven day backward asof
NUTRIENT_ASOF_NAME = 'nerrs.nutrients_asof_7d_hourly.csv'
# quick station level diagnostics
STATION_REPORT_NAME = 'nerrs.station_coverage_report.csv'

# accepted qaqc flags for usable records
GOOD_FLAGS = { 0, 4, -3 }

# filename format station plus type plus optional platform suffix
FILENAME_PATTERN = re.compile( 
    r'^(?P<station>[a-z]{5})(?P<file_type>wq|nut|met)(?:-(?P<platform>[a-z0-9]+))?$',
    re.IGNORECASE,
)

QAQC_PATTERN = re.compile( r'<\s*(?P<flag>-?\d+)\s*>' )

# input datetime formats seen in nerrs files
DATETIME_FORMATS = ( 
    '%m/%d/%Y %H:%M',
    '%m/%d/%Y %H:%M:%S',
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d %H:%M',
)

WQ_RULES = ( 
    ( 'Temp', 'F_Temp', 'w_temp_c' ),
    ( 'Sal', 'F_Sal', 'w_sal_psu' ),
    ( 'DO_mgl', 'F_DO_mgl', 'w_do_mg_l' ),
    ( 'DO_Pct', 'F_DO_Pct', 'w_do_pct' ),
    ( 'Depth', 'F_Depth', 'depth_m' ),
    ( 'pH', 'F_pH', 'w_ph' ),
)

NUT_RULES = ( 
    ( 'PO4F', 'F_PO4F', 'n_po4f_mg_l' ),
    ( 'NH4F', 'F_NH4F', 'n_nh4f_mg_l' ),
    ( 'NO2F', 'F_NO2F', 'n_no2f_mg_l' ),
    ( 'NO3F', 'F_NO3F', 'n_no3f_mg_l' ),
    ( 'NO23F', 'F_NO23F', 'n_no23f_mg_l' ),
    ( 'CHLA_N', 'F_CHLA_N', 'n_chla_ug_l' ),
)

MET_RULES = ( 
    ( 'ATemp', 'F_ATemp', 'm_air_temp_c' ),
    ( 'WSpd', 'F_WSpd', 'm_wind_ms' ),
    ( 'TotPrcp', 'F_TotPrcp', 'm_precip_mm' ),
    ( 'TotSoRad', 'F_TotSoRad', 'm_solar_w_m2' ),
    ( 'RH', 'F_RH', 'm_rh_pct' ),
    ( 'BP', 'F_BP', 'm_bp_hpa' ),
)

WQ_COLUMNS = [ rule[ 2 ] for rule in WQ_RULES ]
NUT_COLUMNS = [ rule[ 2 ] for rule in NUT_RULES ]
MET_COLUMNS = [ rule[ 2 ] for rule in MET_RULES ]



@dataclass( frozen = True )



class StationFile:
    # one discovered csv file and parsed identity
    path: Path
    station: str
    region: str
    file_type: str
    platform: str


def build_parser( ) -> argparse.ArgumentParser:
    # small cli so folk can point at new data drops
    parser = argparse.ArgumentParser( 
        description = 'Build hourly collated NERRS data with nutrient/met as-of expansion.',
    )
    parser.add_argument( 
        '--data-dir',
        type = Path,
        default = DEFAULT_DATA_DIR,
        help = 'Directory containing NERRS station CSV files.',
    )
    parser.add_argument( 
        '--output-dir',
        type = Path,
        default = DEFAULT_OUTPUT_DIR,
        help = 'Directory for collated NERRS output CSV files.',
    )
    parser.add_argument( 
        '--sqlite-path',
        type = Path,
        default = DEFAULT_SQLITE_PATH,
        help = 'Path to transient SQLite merge database.',
    )
    parser.add_argument( 
        '--batch-size',
        type = int,
        default = 5000,
        help = 'SQLite insert batch size.',
    )
    parser.add_argument( 
        '--keep-sqlite',
        action = 'store_true',
        help = 'Keep SQLite file after successful run.',
    )
    return parser


def parse_datetime( raw_value: str ) -> str | None:
    # normalize all timestamps to one sqlite friendly shape
    if raw_value is None:
        return None

    value = raw_value.strip( )
    if not value:
        return None

    for fmt in DATETIME_FORMATS:
        try:
            dt = datetime.strptime( value, fmt )
            return dt.strftime( '%Y-%m-%d %H:%M:%S' )

        except ValueError:
            continue

    return None


def parse_float( raw_value: str | None ) -> float | None:
    # keep numeric parsing strict and drop bad values
    if raw_value is None:
        return None

    value = raw_value.strip( )
    if not value:
        return None

    value = value.replace( ',', '' )

    try:
        return float( value )

    except ValueError:
        return None


def extract_flag( raw_value: str | None ) -> int | None:
    # pull numeric flag from either raw int or <n> style token
    if raw_value is None:
        return None

    value = raw_value.strip( )
    if not value:
        return None

    match = QAQC_PATTERN.search( value )
    if match:
        try:
            return int( match.group( 'flag' ) )

        except ValueError:
            return None

    try:
        return int( value )

    except ValueError:
        return None


def is_observation_csv( file_path: Path ) -> bool:
    # filter out support files like station docs
    try:
        with open( file_path, 'r', encoding = 'utf-8', errors = 'ignore' ) as handle:
            reader = csv.reader( handle )
            headers = next( reader, [ ] )

    except Exception:
        return False

    header_set = { h.strip( ) for h in headers if h }
    has_station = 'StationCode' in header_set
    has_datetime = 'DateTimeStamp' in header_set or 'DatetimeStamp' in header_set

    return has_station and has_datetime


def classify_station_filename( stem: str ) -> dict[ str, str ] | None:
    # derive station region type platform from file stem
    match = FILENAME_PATTERN.match( stem.lower( ) )
    if not match:
        return None

    station = match.group( 'station' )
    file_type = match.group( 'file_type' )
    platform = match.group( 'platform' ) or ''

    return { 
        'station': station,
        'region': station[ :3 ],
        'file_type': file_type,
        'platform': platform,
    }


def discover_station_files( data_dir: Path ) -> list[ StationFile ]:
    # scan for csv files that match station naming and headers
    if not data_dir.exists( ):
        raise FileNotFoundError( f'missing data directory: {data_dir}' )

    discovered: list[ StationFile ] = [ ]
    skipped_non_matching = 0
    skipped_non_observation = 0

    for csv_path in sorted( data_dir.rglob( '*.csv' ) ):
        classification = classify_station_filename( csv_path.stem )
        if classification is None:
            skipped_non_matching += 1
            continue

        if not is_observation_csv( csv_path ):
            skipped_non_observation += 1
            continue

        discovered.append( 
            StationFile( 
                path = csv_path,
                station = classification[ 'station' ],
                region = classification[ 'region' ],
                file_type = classification[ 'file_type' ],
                platform = classification[ 'platform' ],
            ),
        )

    print( f'Found {len( discovered )} station observation files.' )
    print( f'Skipped {skipped_non_matching} files with non-station naming.' )
    print( f'Skipped {skipped_non_observation} files without StationCode/DateTimeStamp headers.' )

    type_counts = { 'wq': 0, 'nut': 0, 'met': 0 }
    for station_file in discovered:
        type_counts[ station_file.file_type ] += 1

    print( f"File counts by type: WQ={type_counts[ 'wq' ]}, NUT={type_counts[ 'nut' ]}, MET={type_counts[ 'met' ]}" )

    return discovered


def build_station_manifest( files: Iterable[ StationFile ] ) -> tuple[ set[ str ], dict[ str, set[ str ] ], bool ]:
    # station eligibility is file presence not timestamp overlap
    station_types: dict[ str, set[ str ] ] = { }

    for station_file in files:
        station_types.setdefault( station_file.station, set( ) ).add( station_file.file_type )

    eligible_stations = { 
        station
        for station, present_types in station_types.items( )
        if 'wq' in present_types and 'nut' in present_types
    }

    met_available = any( 'met' in present_types for present_types in station_types.values( ) )

    print( f'Unique stations discovered: {len( station_types )}' )
    print( f'Eligible stations (WQ + NUT file presence): {len( eligible_stations )}' )
    print( f'MET files present at all: {met_available}' )

    return eligible_stations, station_types, met_available


def initialize_sqlite( sqlite_path: Path ) -> sqlite3.Connection:
    # sqlite is the merge engine so we can process large drops safely
    sqlite_path.parent.mkdir( parents = True, exist_ok = True )

    if sqlite_path.exists( ):
        sqlite_path.unlink( )

    connection = sqlite3.connect( sqlite_path )
    # pragmas tuned for fast local batch writes
    connection.execute( 'PRAGMA journal_mode = WAL' )
    connection.execute( 'PRAGMA synchronous = NORMAL' )
    connection.execute( 'PRAGMA temp_store = MEMORY' )
    connection.execute( 'PRAGMA cache_size = -200000' )

    connection.executescript( 
        '''
        CREATE TABLE IF NOT EXISTS eligible_stations (
            region TEXT NOT NULL,
            station TEXT NOT NULL,
            PRIMARY KEY (station)
        );

        CREATE TABLE IF NOT EXISTS wq_obs (
            region TEXT NOT NULL,
            station TEXT NOT NULL,
            datetime_hour TEXT NOT NULL,
            col_name TEXT NOT NULL,
            value REAL NOT NULL,
            source_file TEXT NOT NULL,
            PRIMARY KEY (region, station, datetime_hour, col_name)
        );

        CREATE TABLE IF NOT EXISTS nut_obs (
            region TEXT NOT NULL,
            station TEXT NOT NULL,
            obs_datetime TEXT NOT NULL,
            col_name TEXT NOT NULL,
            value REAL NOT NULL,
            source_file TEXT NOT NULL,
            PRIMARY KEY (region, station, obs_datetime, col_name)
        );

        CREATE TABLE IF NOT EXISTS met_obs (
            region TEXT NOT NULL,
            station TEXT NOT NULL,
            obs_datetime TEXT NOT NULL,
            col_name TEXT NOT NULL,
            value REAL NOT NULL,
            source_file TEXT NOT NULL,
            PRIMARY KEY (region, station, obs_datetime, col_name)
        );

        CREATE INDEX IF NOT EXISTS idx_wq_station_time
            ON wq_obs (station, datetime_hour, col_name);

        CREATE INDEX IF NOT EXISTS idx_nut_station_time
            ON nut_obs (station, obs_datetime, col_name);

        CREATE INDEX IF NOT EXISTS idx_met_station_time
            ON met_obs (station, obs_datetime, col_name);

        CREATE INDEX IF NOT EXISTS idx_met_region_time
            ON met_obs (region, obs_datetime, col_name);
        ''',
    )

    return connection


def add_column_if_missing( connection: sqlite3.Connection, table_name: str, column_name: str, column_type: str ) -> None:
    # baseline table grows as we add asof metadata columns
    existing = { 
        row[ 1 ]
        for row in connection.execute( f'PRAGMA table_info( {table_name} )' )
    }

    if column_name in existing:
        return

    connection.execute( f'ALTER TABLE {table_name} ADD COLUMN "{column_name}" {column_type}' )


def extract_clean_value( 
    row: dict[ str, str ],
    raw_col: str,
    flag_col: str,
) -> float | None:
    # value must be numeric and pass approved qaqc flags
    value = parse_float( row.get( raw_col ) )
    if value is None:
        return None

    if flag_col in row:
        raw_flag = row.get( flag_col, '' ).strip( )
        if raw_flag:
            parsed_flag = extract_flag( raw_flag )
            if parsed_flag is None or parsed_flag not in GOOD_FLAGS:
                return None

    return value


def ingest_station_files( 
    connection: sqlite3.Connection,
    station_files: Iterable[ StationFile ],
    eligible_stations: set[ str ],
    *,
    batch_size: int,
) -> dict[ str, int ]:
    # read csv rows then write normalized observation records
    stats = { 
        'files_ingested': 0,
        'files_skipped_ineligible': 0,
        'rows_scanned': 0,
        'rows_unparsed_datetime': 0,
        'rows_wq_not_hour': 0,
        'wq_records_inserted': 0,
        'nut_records_inserted': 0,
        'met_records_inserted': 0,
    }

    insert_sql = { 
        'wq': ( 
            'INSERT OR IGNORE INTO wq_obs '
            '( region, station, datetime_hour, col_name, value, source_file ) '
            'VALUES ( ?, ?, ?, ?, ?, ? )'
        ),
        'nut': ( 
            'INSERT OR IGNORE INTO nut_obs '
            '( region, station, obs_datetime, col_name, value, source_file ) '
            'VALUES ( ?, ?, ?, ?, ?, ? )'
        ),
        'met': ( 
            'INSERT OR IGNORE INTO met_obs '
            '( region, station, obs_datetime, col_name, value, source_file ) '
            'VALUES ( ?, ?, ?, ?, ?, ? )'
        ),
    }

    rules_by_type = { 
        'wq': WQ_RULES,
        'nut': NUT_RULES,
        'met': MET_RULES,
    }

    for station_file in station_files:
        # wq and nut only for eligible stations
        if station_file.file_type in { 'wq', 'nut' } and station_file.station not in eligible_stations:
            stats[ 'files_skipped_ineligible' ] += 1
            continue

        records: list[ tuple[ str, str, str, str, float, str ] ] = [ ]
        rules = rules_by_type[ station_file.file_type ]

        with open( station_file.path, 'r', encoding = 'utf-8', errors = 'ignore' ) as handle:
            reader = csv.DictReader( handle )

            for row in reader:
                stats[ 'rows_scanned' ] += 1

                # keep one datetime standard across all tables
                raw_datetime = row.get( 'DateTimeStamp' ) or row.get( 'DatetimeStamp' )
                normalized_datetime = parse_datetime( raw_datetime or '' )

                if normalized_datetime is None:
                    stats[ 'rows_unparsed_datetime' ] += 1
                    continue

                if station_file.file_type == 'wq':
                    # baseline grid is hourly only minute zero
                    parsed_dt = datetime.strptime( normalized_datetime, '%Y-%m-%d %H:%M:%S' )
                    if parsed_dt.minute != 0 or parsed_dt.second != 0:
                        stats[ 'rows_wq_not_hour' ] += 1
                        continue

                for raw_col, flag_col, out_col in rules:
                    clean_value = extract_clean_value( row, raw_col, flag_col )
                    if clean_value is None:
                        continue

                    records.append( 
                        ( 
                            station_file.region,
                            station_file.station,
                            normalized_datetime,
                            out_col,
                            clean_value,
                            station_file.path.name,
                        ),
                    )

                if len( records ) >= batch_size:
                    # write in batches so inserts stay fast
                    before = connection.total_changes
                    connection.executemany( insert_sql[ station_file.file_type ], records )
                    inserted_now = connection.total_changes - before

                    if station_file.file_type == 'wq':
                        stats[ 'wq_records_inserted' ] += inserted_now

                    elif station_file.file_type == 'nut':
                        stats[ 'nut_records_inserted' ] += inserted_now

                    else:
                        stats[ 'met_records_inserted' ] += inserted_now

                    records.clear( )

        if records:
            before = connection.total_changes
            connection.executemany( insert_sql[ station_file.file_type ], records )
            inserted_now = connection.total_changes - before

            if station_file.file_type == 'wq':
                stats[ 'wq_records_inserted' ] += inserted_now

            elif station_file.file_type == 'nut':
                stats[ 'nut_records_inserted' ] += inserted_now

            else:
                stats[ 'met_records_inserted' ] += inserted_now

        stats[ 'files_ingested' ] += 1

    connection.commit( )

    return stats


def load_eligible_stations_table( connection: sqlite3.Connection, eligible_stations: set[ str ] ) -> None:
    # simple station list used by downstream joins
    connection.execute( 'DELETE FROM eligible_stations' )

    rows = [ ( station[ :3 ], station ) for station in sorted( eligible_stations ) ]
    connection.executemany( 
        'INSERT OR REPLACE INTO eligible_stations ( region, station ) VALUES ( ?, ? )',
        rows,
    )
    connection.commit( )


def create_wq_hourly_baseline( connection: sqlite3.Connection ) -> None:
    # pivot long wq rows into one hourly row per station
    pivot_fields = ',\n            '.join( 
        [ 
            f"MAX( CASE WHEN w.col_name = '{column}' THEN w.value END ) AS \"{column}\""
            for column in WQ_COLUMNS
        ],
    )

    connection.execute( 'DROP TABLE IF EXISTS baseline' )

    connection.execute( 
        f'''
        CREATE TABLE baseline AS
        SELECT
            w.region AS region,
            w.station AS station,
            w.datetime_hour AS datetime,
            {pivot_fields}
        FROM wq_obs AS w
        JOIN eligible_stations AS e
            ON e.station = w.station
        GROUP BY w.region, w.station, w.datetime_hour
        ORDER BY w.region, w.station, w.datetime_hour
        ''',
    )

    connection.execute( 
        'CREATE UNIQUE INDEX IF NOT EXISTS idx_baseline_unique ON baseline ( region, station, datetime )',
    )
    connection.commit( )


def create_nutrient_observed_table( connection: sqlite3.Connection ) -> None:
    # save direct nutrient observations before any fill
    pivot_fields = ',\n            '.join( 
        [ 
            f"MAX( CASE WHEN n.col_name = '{column}' THEN n.value END ) AS \"{column}\""
            for column in NUT_COLUMNS
        ],
    )

    connection.execute( 'DROP TABLE IF EXISTS nutrients_observed_clean' )

    connection.execute( 
        f'''
        CREATE TABLE nutrients_observed_clean AS
        SELECT
            n.region AS region,
            n.station AS station,
            n.obs_datetime AS datetime,
            {pivot_fields}
        FROM nut_obs AS n
        JOIN eligible_stations AS e
            ON e.station = n.station
        GROUP BY n.region, n.station, n.obs_datetime
        ORDER BY n.region, n.station, n.obs_datetime
        ''',
    )

    connection.commit( )


def apply_nutrient_asof( connection: sqlite3.Connection ) -> None:
    # attach nutrient nearest prior value within seven days
    for nutrient_column in NUT_COLUMNS:
        add_column_if_missing( connection, 'baseline', nutrient_column, 'REAL' )
        add_column_if_missing( connection, 'baseline', f'{nutrient_column}_obs_time', 'TEXT' )
        add_column_if_missing( connection, 'baseline', f'{nutrient_column}_age_hours', 'REAL' )
        add_column_if_missing( connection, 'baseline', f'{nutrient_column}_observed_now', 'INTEGER' )
        add_column_if_missing( connection, 'baseline', f'{nutrient_column}_imputed_asof', 'INTEGER' )

    for index, nutrient_column in enumerate( NUT_COLUMNS ):
        # pick one best prior nutrient record per baseline row
        temp_table = f'nutr_best_{index}'
        connection.execute( f'DROP TABLE IF EXISTS {temp_table}' )

        connection.execute( 
            f'''
            CREATE TEMP TABLE {temp_table} AS
            WITH candidates AS (
                SELECT
                    b.rowid AS baseline_id,
                    b.datetime AS baseline_datetime,
                    n.obs_datetime AS obs_datetime,
                    n.value AS value
                FROM baseline AS b
                JOIN nut_obs AS n
                    ON n.station = b.station
                    AND n.col_name = ?
                    AND n.obs_datetime <= b.datetime
                    AND n.obs_datetime >= datetime( b.datetime, '-168 hours' )
            ),
            ranked AS (
                SELECT
                    baseline_id,
                    value,
                    obs_datetime,
                    ( julianday( baseline_datetime ) - julianday( obs_datetime ) ) * 24.0 AS age_hours,
                    ROW_NUMBER( ) OVER (
                        PARTITION BY baseline_id
                        ORDER BY obs_datetime DESC
                    ) AS rank_order
                FROM candidates
            )
            SELECT
                baseline_id,
                value,
                obs_datetime,
                age_hours
            FROM ranked
            WHERE rank_order = 1
            ''',
            ( nutrient_column, ),
        )

        connection.execute( 
            # mark exact hour hits vs imputed asof fills
            f'''
            UPDATE baseline
            SET
                "{nutrient_column}" = (
                    SELECT value
                    FROM {temp_table}
                    WHERE baseline_id = baseline.rowid
                ),
                "{nutrient_column}_obs_time" = (
                    SELECT obs_datetime
                    FROM {temp_table}
                    WHERE baseline_id = baseline.rowid
                ),
                "{nutrient_column}_age_hours" = (
                    SELECT age_hours
                    FROM {temp_table}
                    WHERE baseline_id = baseline.rowid
                )
            ''',
        )

        connection.execute( 
            f'''
            UPDATE baseline
            SET
                "{nutrient_column}_observed_now" = CASE
                    WHEN "{nutrient_column}_age_hours" IS NULL THEN 0
                    WHEN ABS( "{nutrient_column}_age_hours" ) <= 0.000001 THEN 1
                    ELSE 0
                END,
                "{nutrient_column}_imputed_asof" = CASE
                    WHEN "{nutrient_column}_age_hours" IS NULL THEN 0
                    WHEN "{nutrient_column}_age_hours" > 0.000001 THEN 1
                    ELSE 0
                END
            ''',
        )

        connection.execute( f'DROP TABLE IF EXISTS {temp_table}' )

    connection.execute( 'DROP TABLE IF EXISTS nutrients_asof_7d_hourly' )
    # persist nutrient asof table as standalone output

    selected_columns = [ 'region', 'station', 'datetime' ]
    for nutrient_column in NUT_COLUMNS:
        selected_columns.extend( 
            [ 
                nutrient_column,
                f'{nutrient_column}_obs_time',
                f'{nutrient_column}_age_hours',
                f'{nutrient_column}_observed_now',
                f'{nutrient_column}_imputed_asof',
            ],
        )

    select_sql = ', '.join( [ f'"{column}"' for column in selected_columns ] )

    connection.execute( 
        f'''
        CREATE TABLE nutrients_asof_7d_hourly AS
        SELECT {select_sql}
        FROM baseline
        ORDER BY region, station, datetime
        ''',
    )

    connection.commit( )


def apply_meteorology_asof( connection: sqlite3.Connection ) -> None:
    # attach met with one hour backward tolerance
    for met_column in MET_COLUMNS:
        add_column_if_missing( connection, 'baseline', met_column, 'REAL' )
        add_column_if_missing( connection, 'baseline', f'{met_column}_source_region', 'TEXT' )
        add_column_if_missing( connection, 'baseline', f'{met_column}_source_station', 'TEXT' )
        add_column_if_missing( connection, 'baseline', f'{met_column}_source_datetime', 'TEXT' )
        add_column_if_missing( connection, 'baseline', f'{met_column}_age_minutes', 'REAL' )
        add_column_if_missing( connection, 'baseline', f'{met_column}_source_priority', 'INTEGER' )

    met_rows = connection.execute( 'SELECT COUNT( * ) FROM met_obs' ).fetchone( )[ 0 ]
    if met_rows == 0:
        print( 'No MET observations found; MET columns remain NULL.' )
        connection.commit( )
        return

    print( f'MET observations available: {met_rows}' )

    for index, met_column in enumerate( MET_COLUMNS ):
        # fallback order station then region then cross region
        temp_table = f'met_best_{index}'
        connection.execute( f'DROP TABLE IF EXISTS {temp_table}' )

        connection.execute( 
            f'''
            CREATE TEMP TABLE {temp_table} AS
            WITH candidates AS (
                SELECT
                    b.rowid AS baseline_id,
                    b.datetime AS baseline_datetime,
                    m.value AS value,
                    m.region AS src_region,
                    m.station AS src_station,
                    m.obs_datetime AS src_datetime,
                    1 AS priority
                FROM baseline AS b
                JOIN met_obs AS m
                    ON m.col_name = ?
                    AND m.station = b.station
                    AND m.obs_datetime <= b.datetime
                    AND m.obs_datetime >= datetime( b.datetime, '-1 hour' )

                UNION ALL

                SELECT
                    b.rowid AS baseline_id,
                    b.datetime AS baseline_datetime,
                    m.value AS value,
                    m.region AS src_region,
                    m.station AS src_station,
                    m.obs_datetime AS src_datetime,
                    2 AS priority
                FROM baseline AS b
                JOIN met_obs AS m
                    ON m.col_name = ?
                    AND m.region = b.region
                    AND m.station <> b.station
                    AND m.obs_datetime <= b.datetime
                    AND m.obs_datetime >= datetime( b.datetime, '-1 hour' )

                UNION ALL

                SELECT
                    b.rowid AS baseline_id,
                    b.datetime AS baseline_datetime,
                    m.value AS value,
                    m.region AS src_region,
                    m.station AS src_station,
                    m.obs_datetime AS src_datetime,
                    3 AS priority
                FROM baseline AS b
                JOIN met_obs AS m
                    ON m.col_name = ?
                    AND m.region <> b.region
                    AND m.obs_datetime <= b.datetime
                    AND m.obs_datetime >= datetime( b.datetime, '-1 hour' )
            ),
            ranked AS (
                SELECT
                    baseline_id,
                    value,
                    src_region,
                    src_station,
                    src_datetime,
                    priority,
                    ( julianday( baseline_datetime ) - julianday( src_datetime ) ) * 24.0 * 60.0 AS age_minutes,
                    ROW_NUMBER( ) OVER (
                        PARTITION BY baseline_id
                        ORDER BY priority ASC, src_datetime DESC
                    ) AS rank_order
                FROM candidates
            )
            SELECT
                baseline_id,
                value,
                src_region,
                src_station,
                src_datetime,
                priority,
                age_minutes
            FROM ranked
            WHERE rank_order = 1
            ''',
            ( met_column, met_column, met_column ),
        )

        connection.execute( 
            f'''
            UPDATE baseline
            SET
                "{met_column}" = (
                    SELECT value
                    FROM {temp_table}
                    WHERE baseline_id = baseline.rowid
                ),
                "{met_column}_source_region" = (
                    SELECT src_region
                    FROM {temp_table}
                    WHERE baseline_id = baseline.rowid
                ),
                "{met_column}_source_station" = (
                    SELECT src_station
                    FROM {temp_table}
                    WHERE baseline_id = baseline.rowid
                ),
                "{met_column}_source_datetime" = (
                    SELECT src_datetime
                    FROM {temp_table}
                    WHERE baseline_id = baseline.rowid
                ),
                "{met_column}_source_priority" = (
                    SELECT priority
                    FROM {temp_table}
                    WHERE baseline_id = baseline.rowid
                ),
                "{met_column}_age_minutes" = (
                    SELECT age_minutes
                    FROM {temp_table}
                    WHERE baseline_id = baseline.rowid
                )
            ''',
        )

        connection.execute( f'DROP TABLE IF EXISTS {temp_table}' )

    connection.commit( )


def build_final_collated_query( connection: sqlite3.Connection ) -> str:
    # build readable column order for export
    table_cols = [ row[ 1 ] for row in connection.execute( 'PRAGMA table_info( baseline )' ) ]

    ordered_columns: list[ str ] = [ 'region', 'station', 'datetime' ]

    ordered_columns.extend( [ col for col in WQ_COLUMNS if col in table_cols ] )
    ordered_columns.extend( [ col for col in NUT_COLUMNS if col in table_cols ] )
    ordered_columns.extend( [ col for col in MET_COLUMNS if col in table_cols ] )

    for met_column in MET_COLUMNS:
        for suffix in ( 
            'source_region',
            'source_station',
            'source_datetime',
            'age_minutes',
            'source_priority',
        ):
            candidate = f'{met_column}_{suffix}'
            if candidate in table_cols:
                ordered_columns.append( candidate )

    for nutrient_column in NUT_COLUMNS:
        for suffix in ( 
            'obs_time',
            'age_hours',
            'observed_now',
            'imputed_asof',
        ):
            candidate = f'{nutrient_column}_{suffix}'
            if candidate in table_cols:
                ordered_columns.append( candidate )

    quoted = ', '.join( [ f'"{column}"' for column in ordered_columns ] )

    return f'SELECT {quoted} FROM baseline ORDER BY region, station, datetime'


def export_query_to_csv( connection: sqlite3.Connection, query: str, output_path: Path ) -> int:
    # stream query results to csv in chunks
    output_path.parent.mkdir( parents = True, exist_ok = True )

    with open( output_path, 'w', newline = '', encoding = 'utf-8' ) as handle:
        writer = csv.writer( handle )
        cursor = connection.execute( query )
        headers = [ description[ 0 ] for description in cursor.description ]
        writer.writerow( headers )

        row_count = 0
        while True:
            chunk = cursor.fetchmany( 10000 )
            if not chunk:
                break

            writer.writerows( chunk )
            row_count += len( chunk )

    return row_count


def write_station_coverage_report( connection: sqlite3.Connection, report_path: Path ) -> int:
    # summarize row coverage so sparse stations stand out
    query = '''
        SELECT
            e.region,
            e.station,
            (
                SELECT COUNT( DISTINCT w.datetime_hour )
                FROM wq_obs AS w
                WHERE w.station = e.station
            ) AS wq_hourly_rows,
            (
                SELECT COUNT( DISTINCT n.obs_datetime )
                FROM nut_obs AS n
                WHERE n.station = e.station
            ) AS nutrient_observation_rows,
            (
                SELECT COUNT( DISTINCT m.obs_datetime )
                FROM met_obs AS m
                WHERE m.station = e.station
            ) AS met_observation_rows_station,
            (
                SELECT COUNT( DISTINCT m.obs_datetime )
                FROM met_obs AS m
                WHERE m.region = e.region
            ) AS met_observation_rows_region
        FROM eligible_stations AS e
        ORDER BY e.region, e.station
    '''

    return export_query_to_csv( connection, query, report_path )


def run_filename_parser_checks( ) -> None:
    # guard against silent parser regressions
    expectations = { 
        'lksbawq-p': ( 'lks', 'lksba', 'wq' ),
        'lksblnut-p': ( 'lks', 'lksbl', 'nut' ),
        'lkspomet-p': ( 'lks', 'lkspo', 'met' ),
        'rkbfbwq-p': ( 'rkb', 'rkbfb', 'wq' ),
        'rkbpbnut-s': ( 'rkb', 'rkbpb', 'nut' ),
        'rkbhcmet-p': ( 'rkb', 'rkbhc', 'met' ),
    }

    for sample, expected in expectations.items( ):
        parsed = classify_station_filename( sample )
        if parsed is None:
            raise ValueError( f'filename parser failed for sample: {sample}' )

        observed = ( parsed[ 'region' ], parsed[ 'station' ], parsed[ 'file_type' ] )
        if observed != expected:
            raise ValueError( f'filename parser mismatch for {sample}: expected {expected}, got {observed}' )

    print( 'Filename parser checks passed for supplied naming examples.' )


def run_integrity_checks( connection: sqlite3.Connection ) -> None:
    # hard checks before writing final files
    duplicate_keys = connection.execute( 
        '''
        SELECT COUNT( * )
        FROM (
            SELECT region, station, datetime, COUNT( * ) AS c
            FROM baseline
            GROUP BY region, station, datetime
            HAVING c > 1
        )
        ''',
    ).fetchone( )[ 0 ]

    if duplicate_keys:
        raise ValueError( f'baseline has duplicate region/station/datetime keys: {duplicate_keys}' )

    leaked_rows = connection.execute( 
        '''
        SELECT COUNT( * )
        FROM baseline AS b
        LEFT JOIN eligible_stations AS e
            ON e.station = b.station
        WHERE e.station IS NULL
        ''',
    ).fetchone( )[ 0 ]

    if leaked_rows:
        raise ValueError( f'baseline contains ineligible station rows: {leaked_rows}' )

    for nutrient_column in NUT_COLUMNS:
        max_age = connection.execute( 
            f'SELECT MAX( "{nutrient_column}_age_hours" ) FROM nutrients_asof_7d_hourly',
        ).fetchone( )[ 0 ]

        if max_age is not None and max_age > 168.000001:
            raise ValueError( f'{nutrient_column} has age_hours > 168 ({max_age})' )

        bad_flag_rows = connection.execute( 
            f'''
            SELECT COUNT( * )
            FROM nutrients_asof_7d_hourly
            WHERE
                "{nutrient_column}_age_hours" IS NULL
                AND (
                    "{nutrient_column}_observed_now" <> 0
                    OR "{nutrient_column}_imputed_asof" <> 0
                )
            ''',
        ).fetchone( )[ 0 ]

        if bad_flag_rows:
            raise ValueError( f'{nutrient_column} has inconsistent observed/imputed flags on NULL age' )

    for met_column in MET_COLUMNS:
        max_met_age = connection.execute( 
            f'SELECT MAX( "{met_column}_age_minutes" ) FROM baseline',
        ).fetchone( )[ 0 ]

        if max_met_age is not None and max_met_age > 60.000001:
            raise ValueError( f'{met_column} has age_minutes > 60 ({max_met_age})' )

    nutrient_any = ' OR '.join( [ f'"{column}" IS NOT NULL' for column in NUT_COLUMNS ] )
    met_any = ' OR '.join( [ f'"{column}" IS NOT NULL' for column in MET_COLUMNS ] )

    summary_query = f'''
        SELECT
            region,
            station,
            COUNT( * ) AS row_count,
            100.0 * AVG( CASE WHEN {nutrient_any} THEN 1 ELSE 0 END ) AS pct_rows_with_any_nutrient,
            100.0 * AVG( CASE WHEN {met_any} THEN 1 ELSE 0 END ) AS pct_rows_with_any_met
        FROM baseline
        GROUP BY region, station
        ORDER BY region, station
    '''

    print( '\nRegion/station collated summary:' )
    for row in connection.execute( summary_query ):
        region, station, row_count, pct_nutr, pct_met = row
        print( 
            f'  {region}-{station}: rows={row_count}, '
            f'pct_rows_with_any_nutrient={pct_nutr:.2f}, '
            f'pct_rows_with_any_met={pct_met:.2f}',
        )


def cleanup_sqlite_files( sqlite_path: Path ) -> None:
    # remove sqlite plus wal and shm sidecar files
    for suffix in ( '', '-wal', '-shm' ):
        candidate = Path( f'{sqlite_path}{suffix}' )
        if candidate.exists( ):
            candidate.unlink( )


def main( ) -> None:
    # top level flow discover ingest merge validate export
    args = build_parser( ).parse_args( )

    data_dir = args.data_dir.resolve( )
    output_dir = args.output_dir.resolve( )
    sqlite_path = args.sqlite_path.resolve( )

    output_dir.mkdir( parents = True, exist_ok = True )

    print( '=' * 72 )
    print( 'Hourly NERRS Collation + Nutrient/MET As-Of Pipeline' )
    print( '=' * 72 )
    print( f'Data directory: {data_dir}' )
    print( f'Output directory: {output_dir}' )
    print( f'SQLite path: {sqlite_path}' )
    print( 'Note: ERA5 merge is intentionally out-of-scope for this script.' )

    run_filename_parser_checks( )

    # phase one discovery and eligibility
    station_files = discover_station_files( data_dir )
    if not station_files:
        raise ValueError( 'No station observation files discovered.' )

    eligible_stations, station_types, met_available = build_station_manifest( station_files )
    if not eligible_stations:
        raise ValueError( 'No eligible stations found with both WQ and NUT file presence.' )

    print( f'Manifest stations tracked: {len( station_types )}' )

    connection = initialize_sqlite( sqlite_path )

    try:
        # phase two ingest raw observations
        load_eligible_stations_table( connection, eligible_stations )

        ingest_stats = ingest_station_files( 
            connection,
            station_files,
            eligible_stations,
            batch_size = args.batch_size,
        )

        print( '\nIngestion stats:' )
        for key, value in ingest_stats.items( ):
            print( f'  {key}: {value}' )

        # phase three build collated tables
        create_wq_hourly_baseline( connection )
        create_nutrient_observed_table( connection )
        apply_nutrient_asof( connection )

        if met_available:
            apply_meteorology_asof( connection )

        else:
            print( 'No MET files in manifest; MET enrichment skipped.' )
            apply_meteorology_asof( connection )

        # phase four validation then exports
        run_integrity_checks( connection )

        final_output = output_dir / FINAL_OUTPUT_NAME
        observed_output = output_dir / NUTRIENT_OBSERVED_NAME
        asof_output = output_dir / NUTRIENT_ASOF_NAME
        report_output = output_dir / STATION_REPORT_NAME

        final_query = build_final_collated_query( connection )
        final_rows = export_query_to_csv( connection, final_query, final_output )
        observed_rows = export_query_to_csv( 
            connection,
            'SELECT * FROM nutrients_observed_clean ORDER BY region, station, datetime',
            observed_output,
        )
        asof_rows = export_query_to_csv( 
            connection,
            'SELECT * FROM nutrients_asof_7d_hourly ORDER BY region, station, datetime',
            asof_output,
        )
        report_rows = write_station_coverage_report( connection, report_output )

        print( '\nOutput files written:' )
        print( f'  {final_output} (rows={final_rows})' )
        print( f'  {observed_output} (rows={observed_rows})' )
        print( f'  {asof_output} (rows={asof_rows})' )
        print( f'  {report_output} (rows={report_rows})' )

    finally:
        connection.close( )

        if args.keep_sqlite:
            print( f'Keeping SQLite database: {sqlite_path}' )

        else:
            cleanup_sqlite_files( sqlite_path )
            print( f'Removed transient SQLite files rooted at: {sqlite_path}' )



if __name__ == '__main__':
    main( )
