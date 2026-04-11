from datetime import datetime, timezone
from pathlib import Path

import joblib


def build_t4d_model_bundle( 
    *,
    domain_feature_cols = None,
    scaler_domain = None,
    kmeans_model = None,
    cluster_name_map = None,
    cluster_order = None,
    cluster_name_order = None,
    cluster_label_order = None,
    cluster_note_order = None,
    phase9_regime_feature_cols = None,
    centroids_z = None,
    feature_mean = None,
    feature_scale = None,
    phase6_selected_model = None,
    phase6_selected_features = None,
    phase6_selected_fill_values = None,
    phase6_context_table = None,
    phase7_targets = None,
    phase7_model_store = None,
    phase7_feature_store = None,
    phase7_fill_store = None,
    phase7_station_context = None,
):
    return { 
        'bundle_name': 't4d_estuary_model_bundle',
        'bundle_version': '2026-04-10',
        'saved_utc': datetime.now( timezone.utc ).isoformat( ),
        'baseline_regime_model': { 
            'domain_feature_cols': list( domain_feature_cols ) if domain_feature_cols is not None else None,
            'scaler_domain': scaler_domain,
            'kmeans_model': kmeans_model,
            'cluster_name_map': cluster_name_map,
            'cluster_order': list( cluster_order ) if cluster_order is not None else None,
            'cluster_name_order': list( cluster_name_order ) if cluster_name_order is not None else None,
            'cluster_label_order': list( cluster_label_order ) if cluster_label_order is not None else None,
            'cluster_note_order': list( cluster_note_order ) if cluster_note_order is not None else None,
        },
        'rolling_regime_classifier': { 
            'phase9_regime_feature_cols': list( phase9_regime_feature_cols ) if phase9_regime_feature_cols is not None else None,
            'centroids_z': centroids_z,
            'feature_mean': feature_mean,
            'feature_scale': feature_scale,
        },
        'phase6_air_to_water_temp': { 
            'selected_model': phase6_selected_model,
            'selected_features': list( phase6_selected_features ) if phase6_selected_features is not None else None,
            'selected_fill_values': phase6_selected_fill_values,
            'context_table': phase6_context_table,
        },
        'phase7_signature_models': { 
            'targets': list( phase7_targets ) if phase7_targets is not None else None,
            'model_store': phase7_model_store,
            'feature_store': phase7_feature_store,
            'fill_store': phase7_fill_store,
            'station_context': phase7_station_context,
        },
    }


def save_t4d_model_bundle( bundle, out_path ):
    out_path = Path( out_path )
    out_path.parent.mkdir( parents = True, exist_ok = True )
    joblib.dump( bundle, out_path, compress = 3 )

    return { 
        'path': str( out_path.resolve( ) ),
        'bytes': int( out_path.stat( ).st_size ),
        'megabytes': float( out_path.stat( ).st_size / ( 1024 ** 2 ) ),
    }
