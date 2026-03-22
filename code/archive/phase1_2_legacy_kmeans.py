"""
Legacy station-character clustering workflow archived from t4d.full.project.ipynb.

This keeps the earlier 1.2 / correlation / 1.2b experiments for posterity.
Expected in-memory inputs before running this file interactively:
- station_baseline
- station_baseline_display ( optional but used for display merges )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


required_inputs = [ 'station_baseline' ]
missing_inputs = [ name for name in required_inputs if name not in globals( ) ]

if missing_inputs:
    raise RuntimeError( f'missing required in-memory tables: {missing_inputs}' )


# --- legacy 1.2: full feature set ---
# local import fallback so this cell runs even if imports were not re-run
from sklearn.metrics import silhouette_score

# use a small, readable set of station-character features
feature_cols = [ 
    'mean_annual_water_temp',
    'mean_annual_salinity',
    'mean_annual_oxygen',
    'mean_annual_saturation',
    'mean_annual_ph',
    'mean_annual_depth',
    'seasonal_amp_water_temp',
    'seasonal_amp_salinity',
    'seasonal_amp_oxygen',
    'seasonal_amp_saturation',
    'seasonal_amp_ph',
    'seasonal_amp_depth',
    'iqr_depth',
]

kmeans_input = station_baseline[ [ 'region', 'station' ] + feature_cols ].copy( )

# simple missing-value fill: region median, then global median
for col in feature_cols:
    kmeans_input[ col ] = kmeans_input.groupby( 'region' )[ col ].transform( lambda s: s.fillna( s.median( ) ) )
    kmeans_input[ col ] = kmeans_input[ col ].fillna( kmeans_input[ col ].median( ) )

# scale so no one feature dominates by units
scaler = StandardScaler( )
X_scaled = scaler.fit_transform( kmeans_input[ feature_cols ] )

# report elbow and silhouette for a simple K-range
k_scan_rows = [ ]

for k in range( 2, 11 ):
    km_scan = KMeans( n_clusters = k, random_state = 42, n_init = 20 )
    labels_scan = km_scan.fit_predict( X_scaled )

    k_scan_rows.append( { 
        'k': k,
        'inertia': float( km_scan.inertia_ ),
        'silhouette': float( silhouette_score( X_scaled, labels_scan ) ),
    } )

k_scan = pd.DataFrame( k_scan_rows )
k_scan[ 'inertia_drop' ] = k_scan[ 'inertia' ].shift( 1 ) - k_scan[ 'inertia' ]
k_scan[ 'inertia_drop_pct' ] = k_scan[ 'inertia_drop' ] / k_scan[ 'inertia' ].shift( 1 )

print( 'k scan ( elbow + silhouette ):' )
print( k_scan.round( 4 ) )

plt.figure( figsize = ( 10, 4 ) )
plt.plot( k_scan[ 'k' ], k_scan[ 'inertia' ], marker = 'o' )
plt.title( 'Elbow Plot: K vs Inertia' )
plt.xlabel( 'K' )
plt.ylabel( 'Inertia' )
plt.tight_layout( )
plt.show( )

plt.figure( figsize = ( 10, 4 ) )
plt.plot( k_scan[ 'k' ], k_scan[ 'silhouette' ], marker = 'o' )
plt.title( 'K vs Silhouette Score' )
plt.xlabel( 'K' )
plt.ylabel( 'Silhouette' )
plt.tight_layout( )
plt.show( )

# keep this simple and explicit for now
k_clusters = 4
kmeans_model = KMeans( n_clusters = k_clusters, random_state = 42, n_init = 20 )
kmeans_input[ 'cluster' ] = kmeans_model.fit_predict( X_scaled )

# update station tables (drop old cluster if this cell gets re-run)
if 'cluster' in station_baseline.columns:
    station_baseline = station_baseline.drop( columns = [ 'cluster' ] )

station_baseline = station_baseline.merge( 
    kmeans_input[ [ 'region', 'station', 'cluster' ] ],
    on = [ 'region', 'station' ],
    how = 'left',
)

if 'cluster' in station_baseline_display.columns:
    station_baseline_display = station_baseline_display.drop( columns = [ 'cluster' ] )

station_baseline_display = station_baseline_display.merge( 
    kmeans_input[ [ 'region', 'station', 'cluster' ] ],
    on = [ 'region', 'station' ],
    how = 'left',
)

print( 'cluster sizes:' )
print( station_baseline[ 'cluster' ].value_counts( ).sort_index( ) )

cluster_profile = station_baseline.groupby( 'cluster' )[ feature_cols ].mean( ).round( 3 )

station_baseline_display[ [ 
    'region',
    'station',
    'station_name',
    'cluster',
    'baseline_start_year',
    'baseline_end_year',
    'n_valid_years',
] ].sort_values( [ 'cluster', 'region', 'station' ] ).head( 40 )

# quick 2d plot so we can see cluster separation
pca = PCA( n_components = 2 )
pcs = pca.fit_transform( X_scaled )

cluster_plot = kmeans_input[ [ 'region', 'station', 'cluster' ] ].copy( )
cluster_plot[ 'pc1' ] = pcs[ :, 0 ]
cluster_plot[ 'pc2' ] = pcs[ :, 1 ]

plt.figure( figsize = ( 11, 7 ) )
sns.scatterplot( 
    data = cluster_plot,
    x = 'pc1',
    y = 'pc2',
    hue = 'cluster',
    palette = 'tab10',
    s = 70,
    alpha = 0.9,
)
plt.title( 'KMeans Clusters of Station Character ( PCA View )' )
plt.xlabel( 'PC1' )
plt.ylabel( 'PC2' )
plt.tight_layout( )
plt.show( )

# cluster mean feature profiles
plt.figure( figsize = ( 14, 6 ) )
sns.heatmap( cluster_profile, cmap = 'YlGnBu', annot = True, fmt = '.2f' )
plt.title( 'Cluster Mean Station-Character Features' )
plt.xlabel( 'Features' )
plt.ylabel( 'Cluster' )
plt.tight_layout( )
plt.show( )


# --- legacy correlation matrix on full feature set ---
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# correlation matrix for the exact features used in clustering
feature_corr = kmeans_input[ feature_cols ].corr( )

# upper-triangle mask so we only show one half of the matrix
tri_mask = np.triu( np.ones_like( feature_corr, dtype = bool ), k = 1 )

plt.figure( figsize = ( 14, 11 ) )
sns.heatmap( 
    feature_corr,
    mask = tri_mask,
    annot = True,
    fmt = '.2f',
    cmap = 'coolwarm',
    center = 0,
    vmin = -1,
    vmax = 1,
    square = True,
    linewidths = 0.35,
    annot_kws = { 'size': 8 },
    cbar_kws = { 'label': 'Pearson r', 'shrink': 0.85 },
)
plt.title( 'KMeans Feature Correlation Matrix ( Triangle + Labels )' )
plt.tight_layout( )
plt.show( )

# strongest absolute pairwise correlations (excluding self-pairs)
corr_pairs = ( 
    feature_corr
    .where( np.triu( np.ones( feature_corr.shape ), k = 1 ).astype( bool ) )
    .stack( )
    .reset_index( )
    .rename( columns = { 'level_0': 'feature_a', 'level_1': 'feature_b', 0: 'corr' } )
)

corr_pairs[ 'abs_corr' ] = corr_pairs[ 'corr' ].abs( )
corr_pairs = corr_pairs.sort_values( 'abs_corr', ascending = False )

print( 'top correlated feature pairs:' )
corr_pairs.head( 15 )


# --- legacy 1.2b: reduced feature set ---
# second KMeans pass using a simpler feature set
# logic:
# - drop mean_annual_saturation because it tracks mean_annual_oxygen closely
# - drop seasonal_amp_oxygen because it tracks seasonal_amp_water_temp closely
# - keep the rest so we still preserve chemistry + seasonality + depth structure

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

removed_feature_cols = [ 
    'mean_annual_saturation',
    'seasonal_amp_oxygen',
]

reduced_feature_cols = [ col for col in feature_cols if col not in removed_feature_cols ]

print( 'removed features (high-correlation redundancy):', removed_feature_cols )
print( 'reduced feature count:', len( reduced_feature_cols ) )

kmeans_input_reduced = station_baseline[ [ 'region', 'station' ] + reduced_feature_cols ].copy( )

# same simple fill strategy as first pass: region median, then global median
for col in reduced_feature_cols:
    kmeans_input_reduced[ col ] = kmeans_input_reduced.groupby( 'region' )[ col ].transform( lambda s: s.fillna( s.median( ) ) )
    kmeans_input_reduced[ col ] = kmeans_input_reduced[ col ].fillna( kmeans_input_reduced[ col ].median( ) )

scaler_reduced = StandardScaler( )
X_scaled_reduced = scaler_reduced.fit_transform( kmeans_input_reduced[ reduced_feature_cols ] )

# run the same K scan so we can compare shape of elbow/silhouette
k_scan_rows_reduced = [ ]

for k in range( 2, 11 ):
    km_scan = KMeans( n_clusters = k, random_state = 42, n_init = 20 )
    labels_scan = km_scan.fit_predict( X_scaled_reduced )

    k_scan_rows_reduced.append( { 
        'k': k,
        'inertia': float( km_scan.inertia_ ),
        'silhouette': float( silhouette_score( X_scaled_reduced, labels_scan ) ),
    } )

k_scan_reduced = pd.DataFrame( k_scan_rows_reduced )
k_scan_reduced[ 'inertia_drop' ] = k_scan_reduced[ 'inertia' ].shift( 1 ) - k_scan_reduced[ 'inertia' ]
k_scan_reduced[ 'inertia_drop_pct' ] = k_scan_reduced[ 'inertia_drop' ] / k_scan_reduced[ 'inertia' ].shift( 1 )

print( )
print( 'reduced-feature k scan:' )
print( k_scan_reduced.round( 4 ) )

plt.figure( figsize = ( 10, 4 ) )
plt.plot( k_scan_reduced[ 'k' ], k_scan_reduced[ 'inertia' ], marker = 'o' )
plt.title( 'Reduced Features: K vs Inertia' )
plt.xlabel( 'K' )
plt.ylabel( 'Inertia' )
plt.tight_layout( )
plt.show( )

plt.figure( figsize = ( 10, 4 ) )
plt.plot( k_scan_reduced[ 'k' ], k_scan_reduced[ 'silhouette' ], marker = 'o' )
plt.title( 'Reduced Features: K vs Silhouette Score' )
plt.xlabel( 'K' )
plt.ylabel( 'Silhouette' )
plt.tight_layout( )
plt.show( )

# keep K aligned with first pass so comparisons stay straightforward
k_clusters_reduced = k_clusters if 'k_clusters' in globals( ) else 4

kmeans_model_reduced = KMeans( n_clusters = k_clusters_reduced, random_state = 42, n_init = 20 )
kmeans_input_reduced[ 'cluster_reduced' ] = kmeans_model_reduced.fit_predict( X_scaled_reduced )

# compare silhouette at the chosen K between full and reduced feature sets
silhouette_reduced = float( silhouette_score( X_scaled_reduced, kmeans_input_reduced[ 'cluster_reduced' ] ) )

if 'X_scaled' in globals( ):
    kmeans_compare_full = KMeans( n_clusters = k_clusters_reduced, random_state = 42, n_init = 20 )
    labels_full_compare = kmeans_compare_full.fit_predict( X_scaled )
    silhouette_full_compare = float( silhouette_score( X_scaled, labels_full_compare ) )

else:
    silhouette_full_compare = np.nan

print( )
print( f'chosen K ( reduced ): { k_clusters_reduced }' )
print( 'silhouette ( full features ):', round( float( silhouette_full_compare ), 4 ) )
print( 'silhouette ( reduced features ):', round( float( silhouette_reduced ), 4 ) )

print( )
print( 'reduced-feature cluster sizes:' )
print( kmeans_input_reduced[ 'cluster_reduced' ].value_counts( ).sort_index( ) )

# save reduced clusters onto station tables for later inspection
if 'cluster_reduced' in station_baseline.columns:
    station_baseline = station_baseline.drop( columns = [ 'cluster_reduced' ] )

station_baseline = station_baseline.merge( 
    kmeans_input_reduced[ [ 'region', 'station', 'cluster_reduced' ] ],
    on = [ 'region', 'station' ],
    how = 'left',
)

if 'station_baseline_display' in globals( ):
    if 'cluster_reduced' in station_baseline_display.columns:
        station_baseline_display = station_baseline_display.drop( columns = [ 'cluster_reduced' ] )

    station_baseline_display = station_baseline_display.merge( 
        kmeans_input_reduced[ [ 'region', 'station', 'cluster_reduced' ] ],
        on = [ 'region', 'station' ],
        how = 'left',
    )

# quick PCA view of reduced-feature clusters
pca_reduced = PCA( n_components = 2 )
pcs_reduced = pca_reduced.fit_transform( X_scaled_reduced )

cluster_plot_reduced = kmeans_input_reduced[ [ 'region', 'station', 'cluster_reduced' ] ].copy( )
cluster_plot_reduced[ 'pc1' ] = pcs_reduced[ :, 0 ]
cluster_plot_reduced[ 'pc2' ] = pcs_reduced[ :, 1 ]

plt.figure( figsize = ( 11, 7 ) )
sns.scatterplot( 
    data = cluster_plot_reduced,
    x = 'pc1',
    y = 'pc2',
    hue = 'cluster_reduced',
    palette = 'tab10',
    s = 70,
    alpha = 0.9,
)
plt.title( 'Reduced-Feature KMeans Clusters ( PCA View )' )
plt.xlabel( 'PC1' )
plt.ylabel( 'PC2' )
plt.tight_layout( )
plt.show( )

station_baseline[ [ 'region', 'station', 'cluster', 'cluster_reduced' ] ].head( 20 )
