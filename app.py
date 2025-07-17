import pandas as pd
from geopy.distance import geodesic, great_circle
import numpy as np
from sklearn.neighbors import BallTree, NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
import math
import time
from functools import lru_cache
import multiprocessing as mp
from numba import jit, njit
import pickle
import os
warnings.filterwarnings('ignore')

class SunScoreCalculator:
    def __init__(self, file_path='solar_data.csv'):
        """Initialize with optimized data loading and indexing"""
        print("Loading solar data...")
        start_time = time.time()
        
        # Load data in chunks for memory efficiency
        self.df = self._load_data_efficiently(file_path)
        self.df.columns = self.df.columns.str.strip()
        
        # Preprocess data with optimizations
        self._preprocess_data_optimized()
        self._validate_coordinates()
        self._create_advanced_spatial_index()
        
        # Pre-compute expensive calculations
        self._precompute_statistics()
        
        load_time = time.time() - start_time
        print(f"✓ Loaded {len(self.df):,} records in {load_time:.1f}s")
    
    def _load_data_efficiently(self, file_path, chunk_size=50000):
        """Load data in chunks for memory efficiency"""
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Pre-filter obviously invalid data early
            chunk = chunk.dropna(subset=['latitude', 'longitude', 'ghi'])
            chunks.append(chunk)
        
        return pd.concat(chunks, ignore_index=True)
    
    def _preprocess_data_optimized(self):
        """Optimized preprocessing with vectorized operations"""
        # Vectorized timestamp conversion
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        self.df = self.df.dropna(subset=['timestamp'])
        
        # Vectorized date extraction
        self.df['Year'] = self.df['timestamp'].dt.year
        self.df['Month'] = self.df['timestamp'].dt.month
        
        # Optimized zip code processing
        self.df['zip_code'] = (self.df['zip_code'].astype(str)
                              .str.replace('.0', '', regex=False)
                              .str.zfill(5))
        
        # Vectorized coordinate validation
        valid_coords = (
            self.df['latitude'].between(24.396308, 49.384358) &
            self.df['longitude'].between(-125.0, -66.93457) &
            self.df['latitude'].notna() &
            self.df['longitude'].notna()
        )
        self.df = self.df[valid_coords].reset_index(drop=True)
        
        # Optimized location aggregation
        self.locations_df = (self.df.groupby(['latitude', 'longitude', 'zip_code'])
                           .size().reset_index(name='data_count'))
        
        # Vectorized numeric conversion
        self.locations_df['zip_numeric'] = pd.to_numeric(
            self.locations_df['zip_code'], errors='coerce'
        )

    def _validate_coordinates(self):
        """Optimized coordinate validation with spatial clustering"""
        # Use vectorized operations for duplicate detection
        coord_precision = 4
        self.locations_df['coord_hash'] = (
            self.locations_df['latitude'].round(coord_precision).astype(str) + '_' +
            self.locations_df['longitude'].round(coord_precision).astype(str)
        )
        
        # Keep highest data count location per coordinate hash
        self.locations_df = (self.locations_df.sort_values('data_count', ascending=False)
                           .drop_duplicates(subset=['coord_hash'])
                           .drop('coord_hash', axis=1)
                           .reset_index(drop=True))
    
    def _create_advanced_spatial_index(self):
        """Create multiple spatial indices for different query types"""
        if len(self.locations_df) == 0:
            raise ValueError("No valid locations found")
        
        coords = self.locations_df[['latitude', 'longitude']].values
        
        # Primary: BallTree with haversine for geographic queries
        coords_rad = np.radians(coords)
        self.kdtree = BallTree(coords_rad, metric='haversine')
        
        # Secondary: cKDTree for fast euclidean approximations
        self.ckdtree = cKDTree(coords)
        
        # Tertiary: Create spatial grid index for very large datasets
        self._create_grid_index()

    def _create_grid_index(self, grid_size=0.1):
        """Create spatial grid index for ultra-fast lookups on massive datasets"""
        coords = self.locations_df[['latitude', 'longitude']].values
        
        # Create grid boundaries
        lat_min, lat_max = coords[:, 0].min(), coords[:, 0].max()
        lon_min, lon_max = coords[:, 1].min(), coords[:, 1].max()
        
        # Create grid
        lat_bins = np.arange(lat_min, lat_max + grid_size, grid_size)
        lon_bins = np.arange(lon_min, lon_max + grid_size, grid_size)
        
        # Assign points to grid cells
        lat_indices = np.digitize(coords[:, 0], lat_bins)
        lon_indices = np.digitize(coords[:, 1], lon_bins)
        
        # Create grid dictionary
        self.grid_index = {}
        for i, (lat_idx, lon_idx) in enumerate(zip(lat_indices, lon_indices)):
            key = (lat_idx, lon_idx)
            if key not in self.grid_index:
                self.grid_index[key] = []
            self.grid_index[key].append(i)
        
        self.grid_size = grid_size
        self.lat_bins = lat_bins
        self.lon_bins = lon_bins
    
    def _precompute_statistics(self):
        """Pre-compute expensive statistics for faster queries"""
        # Cache zip code statistics
        self.zip_stats = {}
        for zip_code in self.locations_df['zip_code'].unique():
            zip_data = self.df[self.df['zip_code'] == zip_code]
            if len(zip_data) > 0:
                self.zip_stats[zip_code] = {
                    'ghi_mean': zip_data['ghi'].mean(),
                    'ghi_std': zip_data['ghi'].std(),
                    'data_quality': len(zip_data) / len(zip_data.dropna())
                }

    @njit
    def _haversine_distance_numba(self, lat1, lon1, lat2, lon2):
        """Ultra-fast haversine calculation with Numba JIT compilation"""
        R = 6371.0088
        
        lat1_r = math.radians(lat1)
        lon1_r = math.radians(lon1)
        lat2_r = math.radians(lat2)
        lon2_r = math.radians(lon2)
        
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def _get_grid_candidates(self, user_lat, user_lon, search_radius=3):
        """Get candidate locations from grid index for massive datasets"""
        lat_idx = np.digitize(user_lat, self.lat_bins)
        lon_idx = np.digitize(user_lon, self.lon_bins)
        
        candidates = []
        
        # Search in nearby grid cells
        for lat_offset in range(-search_radius, search_radius + 1):
            for lon_offset in range(-search_radius, search_radius + 1):
                key = (lat_idx + lat_offset, lon_idx + lon_offset)
                if key in self.grid_index:
                    candidates.extend(self.grid_index[key])
        
        return candidates
    
    def _advanced_distance_weighting(self, distances, method='inverse_squared'):
        """Advanced distance weighting methods"""
        distances = np.array(distances)
        
        if method == 'inverse_squared':
            # Inverse distance squared weighting
            weights = 1 / (distances**2 + 1e-10)
        elif method == 'gaussian':
            # Gaussian weighting
            sigma = np.mean(distances) / 2
            weights = np.exp(-(distances**2) / (2 * sigma**2))
        elif method == 'exponential':
            # Exponential decay
            weights = np.exp(-distances / np.mean(distances))
        else:
            # Simple inverse distance
            weights = 1 / (distances + 1e-10)
        
        return weights / np.sum(weights)  # Normalize
    
    def _get_nearby_locations_optimized(self, user_lat, user_lon, user_zip=None, k=5, max_distance_km=50):
        """Optimized location search for massive datasets"""
        # Strategy 1: Grid-based pre-filtering for massive datasets
        if len(self.locations_df) > 100000:  # Use grid for very large datasets
            candidate_indices = self._get_grid_candidates(user_lat, user_lon)
            candidate_coords = self.locations_df.iloc[candidate_indices][['latitude', 'longitude']].values
            
            # Fast distance calculation using vectorized operations
            distances = np.array([
                self._haversine_distance_numba(user_lat, user_lon, lat, lon)
                for lat, lon in candidate_coords
            ])
            
            # Filter by distance
            valid_mask = distances <= max_distance_km
            valid_indices = np.array(candidate_indices)[valid_mask]
            valid_distances = distances[valid_mask]

        else:
            # Use BallTree for smaller datasets
            user_coords_rad = np.radians([[user_lat, user_lon]])
            search_k = min(k * 5, len(self.locations_df))
            distances_rad, indices = self.kdtree.query(user_coords_rad, k=search_k)
            
            distances_km = distances_rad[0] * 6371.0088
            valid_mask = distances_km <= max_distance_km
            valid_indices = indices[0][valid_mask]
            valid_distances = distances_km[valid_mask]
        
        if len(valid_indices) == 0:
            return []
        
        # Sort by distance and select top k
        sort_indices = np.argsort(valid_distances)
        top_indices = valid_indices[sort_indices[:k]]
        top_distances = valid_distances[sort_indices[:k]]
        
        # Advanced weighting
        weights = self._advanced_distance_weighting(top_distances, method='inverse_squared')
        
        # Build result with enhanced metadata
        nearby_locations = []
        for i, (idx, distance, weight) in enumerate(zip(top_indices, top_distances, weights)):
            location = self.locations_df.iloc[idx]
            
            # Enhanced match type determination
            match_type = 'spatial'
            if user_zip:
                zip_dist = self._calculate_zip_distance_optimized(user_zip, location['zip_code'])
                if zip_dist == 0:
                    match_type = 'exact_zip'
                elif zip_dist <= 2:
                    match_type = 'close_zip'
            
            nearby_locations.append({
                'zip_code': location['zip_code'],
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'distance': distance,
                'weight': weight,
                'match_type': match_type,
                'data_quality': self.zip_stats.get(location['zip_code'], {}).get('data_quality', 1.0)
            })
        
        return nearby_locations
    
    @lru_cache(maxsize=10000)
    def _calculate_zip_distance_optimized(self, zip1, zip2):
        """Cached zip distance calculation"""
        try:
            zip1_num = int(str(zip1).zfill(5))
            zip2_num = int(str(zip2).zfill(5))
            numeric_diff = abs(zip1_num - zip2_num)
            
            if numeric_diff == 0: return 0
            elif numeric_diff <= 5: return 1
            elif numeric_diff <= 20: return 2
            elif numeric_diff <= 100: return 3
            elif numeric_diff <= 500: return 4
            else: return 5
        except:
            return 10
    
    def _enhanced_interpolation(self, nearby_locations, method='idw'):
        """Advanced interpolation methods for precise estimation"""
        if not nearby_locations:
            return None, 0
        
        # Collect all data with parallel processing
        all_monthly_data = []
        
        def process_location(location):
            zip_code = location['zip_code']
            zip_data = self.df[self.df['zip_code'] == zip_code].copy()
            
            if zip_data.empty:
                return None
            
            # Enhanced monthly aggregation with multiple metrics
            monthly_stats = zip_data.groupby(['Year', 'Month']).agg({
                'ghi': ['mean', 'std', 'count'],
                'dni': ['mean', 'std'],
                'dhi': ['mean', 'std']
            }).reset_index()
            
            # Flatten column names
            monthly_stats.columns = ['_'.join(col).strip('_') for col in monthly_stats.columns]
            
            # Enhanced sunscore with uncertainty quantification
            monthly_stats['raw_sunscore'] = self._calculate_advanced_sunscore(monthly_stats)
            monthly_stats['sunscore_uncertainty'] = self._calculate_uncertainty(monthly_stats)
            monthly_stats['weight'] = location['weight']
            monthly_stats['distance'] = location['distance']
            monthly_stats['zip_code'] = zip_code
            
            return monthly_stats
        
        # Parallel processing for large datasets
        if len(nearby_locations) > 4:
            with ThreadPoolExecutor(max_workers=min(4, len(nearby_locations))) as executor:
                results = list(executor.map(process_location, nearby_locations))
            all_monthly_data = [r for r in results if r is not None]
        else:
            all_monthly_data = [process_location(loc) for loc in nearby_locations if process_location(loc) is not None]
        
        if not all_monthly_data:
            return None, 0
        
        combined_df = pd.concat(all_monthly_data, ignore_index=True)
        
        # Advanced interpolation
        if method == 'idw':
            # Inverse Distance Weighting with uncertainty
            weighted_monthly = combined_df.groupby(['Year', 'Month']).apply(
                lambda x: pd.Series({
                    'Sunscore': np.average(x['raw_sunscore'], weights=x['weight']),
                    'uncertainty': np.sqrt(np.average(x['sunscore_uncertainty']**2, weights=x['weight'])),
                    'data_points': x['ghi_count'].sum(),
                    'avg_distance': np.average(x['distance'], weights=x['weight']),
                    'quality_score': np.average(x['ghi_count'], weights=x['weight'])
                })
            ).reset_index()
        
        # Adaptive normalization with uncertainty bounds
        return self._adaptive_normalization_with_uncertainty(weighted_monthly, nearby_locations)
    
    def _calculate_advanced_sunscore(self, monthly_stats):
        """Advanced sunscore with multiple solar metrics"""
        # Multi-component weighting
        alpha, beta, gamma = 0.6, 0.25, 0.15
        
        base_score = (alpha * monthly_stats['ghi_mean'] + 
                     beta * monthly_stats['dni_mean'] + 
                     gamma * monthly_stats['dhi_mean'])
        
        # Seasonal enhancement
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * (monthly_stats['Month'] - 3) / 12)
        
        # Data quality adjustment
        quality_factor = np.clip(monthly_stats['ghi_count'] / 100, 0.5, 1.2)
        
        return base_score * seasonal_factor * quality_factor
    
    def _calculate_enhanced_sunscore(self, monthly_avg):
        """Enhanced sunscore calculation with seasonal adjustments - wrapper for compatibility"""
        return self._calculate_advanced_sunscore(monthly_avg)
    
    def _calculate_uncertainty(self, monthly_stats):
        """Calculate uncertainty in sunscore estimates"""
        # Uncertainty based on data variability and count
        ghi_uncertainty = monthly_stats['ghi_std'] / np.sqrt(monthly_stats['ghi_count'])
        base_uncertainty = ghi_uncertainty * 0.6  # Scale to sunscore units
        
        # Adjust for data scarcity
        count_factor = 1 / np.sqrt(monthly_stats['ghi_count'] + 1)
        
        return base_uncertainty + count_factor * 10
    
    def _adaptive_normalization_with_uncertainty(self, weighted_monthly, nearby_locations):
        """Adaptive normalization considering uncertainty"""
        if len(weighted_monthly) == 0:
            return None, 0
        
        raw_scores = weighted_monthly['Sunscore']
        uncertainties = weighted_monthly['uncertainty']
        
        print(f"\nAdvanced analysis:")
        print(f"Raw sunscore range: {raw_scores.min():.1f} ± {uncertainties.mean():.1f}")
        print(f"Peak uncertainty: {uncertainties.max():.1f}")
        
        # Determine normalization strategy
        if raw_scores.max() < 500:
            # Use dataset-relative with uncertainty bounds
            q25, q75 = np.percentile(raw_scores, [25, 75])
            iqr = q75 - q25
            min_threshold = max(0, q25 - 1.5 * iqr)
            max_threshold = q75 + 1.5 * iqr
            print(f"Using robust percentile normalization: {min_threshold:.1f} - {max_threshold:.1f}")
        else:
            min_threshold, max_threshold = 500, 2500
        
        # Normalize with uncertainty propagation
        normalized_scores = np.clip(
            100 * (raw_scores - min_threshold) / (max_threshold - min_threshold),
            0, 100
        )
        
        # Propagate uncertainty
        normalized_uncertainty = 100 * uncertainties / (max_threshold - min_threshold)
        
        weighted_monthly['Sunscore'] = normalized_scores
        weighted_monthly['Sunscore_uncertainty'] = normalized_uncertainty
        
        # Enhanced confidence calculation
        confidence = self._calculate_precision_confidence(weighted_monthly, nearby_locations)
        
        return weighted_monthly[['Year', 'Month', 'Sunscore', 'Sunscore_uncertainty']], confidence
    
    def _calculate_precision_confidence(self, monthly_data, nearby_locations):
        """Precision-focused confidence calculation"""
        if monthly_data is None or len(monthly_data) == 0:
            return 0
        
        # Multiple confidence factors
        distances = [loc['distance'] for loc in nearby_locations]
        avg_distance = np.mean(distances)
        min_distance = np.min(distances)
        
        # Distance confidence (exponential decay)
        distance_conf = 100 * np.exp(-avg_distance / 10)
        
        # Precision confidence (based on minimum distance)
        precision_conf = 100 * np.exp(-min_distance / 5)
        
        # Data quality confidence
        avg_quality = np.mean([loc.get('data_quality', 1.0) for loc in nearby_locations])
        quality_conf = avg_quality * 100
        
        # Uncertainty confidence (lower uncertainty = higher confidence)
        avg_uncertainty = monthly_data['Sunscore_uncertainty'].mean()
        uncertainty_conf = max(0, 100 - avg_uncertainty * 2)
        
        # Coverage confidence
        coverage_conf = min(100, len(monthly_data) / 12 * 100)
        
        # Weighted combination emphasizing precision
        confidence = (
            precision_conf * 0.3 +
            distance_conf * 0.25 +
            quality_conf * 0.2 +
            uncertainty_conf * 0.15 +
            coverage_conf * 0.1
        )
        
        return min(100, confidence)
    
    def _calculate_enhanced_confidence_with_zip(self, monthly_data, nearby_locations, user_zip=None):
        """Enhanced confidence calculation considering zip code numeric proximity"""
        if monthly_data is None or len(monthly_data) == 0:
            return 0
        
        # Factor 1: Zip code proximity bonus
        zip_confidence = 30  # Base confidence
        if user_zip:
            # Count exact and close matches
            exact_matches = sum(1 for loc in nearby_locations 
                              if loc.get('zip_distance') == 0 or loc.get('match_type') == 'exact_zip')
            very_close_matches = sum(1 for loc in nearby_locations 
                                   if loc.get('match_type') == 'close_zip')
            
            if exact_matches > 0:
                zip_confidence = min(100, 95 + exact_matches * 2)
            elif very_close_matches > 0:
                zip_confidence = min(100, 85 + very_close_matches * 3)
            else:
                zip_confidence = 40
        
        # Factor 2: Average distance to data points
        avg_distance = np.mean([loc['distance'] for loc in nearby_locations])
        distance_confidence = max(0, 100 - (avg_distance * 2))
        
        # Factor 3: Number of nearby locations used
        location_confidence = min(100, len(nearby_locations) * 25)
        
        # Factor 4: Data coverage
        months_covered = len(monthly_data)
        coverage_confidence = min(100, (months_covered / 12) * 100)
        
        # Factor 5: Data consistency
        if len(monthly_data) > 1:
            sunscore_std = monthly_data['Sunscore'].std()
            consistency_confidence = max(0, 100 - sunscore_std)
        else:
            consistency_confidence = 50
        
        # Factor 6: Data density
        avg_data_points = monthly_data.get('data_points', pd.Series([1])).mean()
        density_confidence = min(100, avg_data_points * 10)
        
        # Weighted combination with higher weight on zip matching
        confidence = (zip_confidence * 0.4 + 
                     distance_confidence * 0.2 + 
                     location_confidence * 0.15 + 
                     coverage_confidence * 0.1 + 
                     consistency_confidence * 0.1 + 
                     density_confidence * 0.05)
        
        return min(100, confidence)

    def get_monthly_sunscore_precision(self, user_lat, user_lon, user_zip=None, k_neighbors=5, max_distance_km=25):
        """Ultra-precise sunscore calculation optimized for massive datasets"""
        # Optimized location search
        nearby_locations = self._get_nearby_locations_optimized(
            user_lat, user_lon, user_zip, k_neighbors, max_distance_km
        )
        
        if not nearby_locations:
            return "No data found within search radius.", 0
        
        # Enhanced interpolation
        result, confidence = self._enhanced_interpolation(nearby_locations, method='idw')
        
        return result, confidence

    def get_monthly_sunscore_precise(self, user_lat, user_lon, user_zip=None, k_neighbors=5, max_distance_km=25):
        """Main method to get monthly sunscore with maximum geographic precision"""
        # Find nearby locations using optimized search
        nearby_locations = self._get_nearby_locations_optimized(
            user_lat, user_lon, user_zip, k_neighbors, max_distance_km
        )
        
        if not nearby_locations:
            # Try with larger radius as fallback
            nearby_locations = self._get_nearby_locations_optimized(
                user_lat, user_lon, user_zip, k_neighbors, max_distance_km * 2
            )
            
            if not nearby_locations:
                return "No data found within reasonable distance.", 0
        
        # Calculate weighted sunscore using enhanced method
        all_monthly_data = []
        total_weight = sum(loc['weight'] for loc in nearby_locations)
        
        for location in nearby_locations:
            zip_code = location['zip_code']
            weight = location['weight'] / total_weight
            
            zip_data = self.df[self.df['zip_code'] == zip_code].copy()
            
            if zip_data.empty:
                continue
            
            monthly_avg = zip_data.groupby(['Year', 'Month']).agg({
                'ghi': 'mean',
                'dni': 'mean',
                'dhi': 'mean'
            }).reset_index()
            
            monthly_avg['raw_sunscore'] = self._calculate_advanced_sunscore_simple(monthly_avg)
            monthly_avg['weight'] = weight
            monthly_avg['zip_code'] = zip_code
            monthly_avg['distance'] = location['distance']
            
            all_monthly_data.append(monthly_avg)
        
        if not all_monthly_data:
            return None, 0
            
        combined_df = pd.concat(all_monthly_data, ignore_index=True)
        
        weighted_monthly = combined_df.groupby(['Year', 'Month']).apply(
            lambda x: pd.Series({
                'Sunscore': np.average(x['raw_sunscore'], weights=x['weight']),
                'data_points': len(x),
                'avg_distance': np.average(x['distance'], weights=x['weight'])
            })
        ).reset_index()
        
        # Adaptive normalization
        if len(weighted_monthly) > 0:
            raw_scores = weighted_monthly['Sunscore']
            min_raw = raw_scores.min()
            max_raw = raw_scores.max()
            
            # Use dataset-relative normalization for this data
            if max_raw > min_raw:
                range_padding = (max_raw - min_raw) * 0.2
                min_threshold = max(0, min_raw - range_padding)
                max_threshold = max_raw + range_padding
            else:
                weighted_monthly['Sunscore'] = 50
                confidence = self._calculate_enhanced_confidence_with_zip(
                    weighted_monthly, nearby_locations, user_zip
                )
                return weighted_monthly[['Year', 'Month', 'Sunscore']], confidence
            
            # Apply normalization
            weighted_monthly['Sunscore'] = np.clip(
                100 * (weighted_monthly['Sunscore'] - min_threshold) / (max_threshold - min_threshold),
                0, 100
            )
        
        # Calculate enhanced confidence
        confidence = self._calculate_enhanced_confidence_with_zip(
            weighted_monthly, nearby_locations, user_zip
        )
        
        # Precision bonus based on average distance
        avg_distance = np.mean([loc['distance'] for loc in nearby_locations])
        if avg_distance < 5:
            confidence += 10
        elif avg_distance < 10:
            confidence += 5
        
        confidence = min(100, confidence)
        
        return weighted_monthly[['Year', 'Month', 'Sunscore']], confidence

    def _calculate_advanced_sunscore_simple(self, monthly_avg):
        """Simple wrapper for compatibility"""
        # Multi-component weighting
        alpha, beta, gamma = 0.6, 0.25, 0.15
        
        base_score = (alpha * monthly_avg['ghi'] + 
                     beta * monthly_avg['dni'] + 
                     gamma * monthly_avg['dhi'])
        
        # Seasonal enhancement
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * (monthly_avg['Month'] - 3) / 12)
        
        return base_score * seasonal_factor

# Streamlined example usage
if __name__ == "__main__":
    # Initialize calculator
    calculator = SunScoreCalculator('solar_data.csv')

    # Test coordinates
    user_lat = 42.04783608530275
    user_lon = -72.62718033091582
    user_zip = "01013"

    print(f"\nAnalyzing solar potential for coordinates: ({user_lat:.4f}, {user_lon:.4f})")
    if user_zip:
        print(f"ZIP Code: {user_zip}")
    
    # Get precise sunscore
    sunscore_data, confidence = calculator.get_monthly_sunscore_precise(
        user_lat, user_lon, 
        user_zip=user_zip,
        k_neighbors=3,
        max_distance_km=15
    )

    # Clean results display
    print("\n" + "="*60)
    print("🌞 SOLAR POTENTIAL ANALYSIS")
    print("="*60)
    
    if sunscore_data is not None and len(sunscore_data) > 0:
        # Monthly breakdown
        print("\n📊 Monthly Solar Scores (0-100 scale):")
        print("-" * 40)
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for _, row in sunscore_data.iterrows():
            month_name = month_names[int(row['Month'])-1]
            score = row['Sunscore']
            
            # Visual bar representation
            bar_length = int(score / 5)  # Scale to 20 chars max
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            print(f"{month_name} {int(row['Year'])}: {score:5.1f} │{bar}│")
        
        # Summary statistics
        annual_avg = sunscore_data['Sunscore'].mean()
        peak_month = sunscore_data.loc[sunscore_data['Sunscore'].idxmax()]
        low_month = sunscore_data.loc[sunscore_data['Sunscore'].idxmin()]
        peak_name = month_names[int(peak_month['Month'])-1]
        low_name = month_names[int(low_month['Month'])-1]
        
        print("\n📈 Key Insights:")
        print("-" * 40)
        print(f"Annual Average Score: {annual_avg:.1f}/100")
        print(f"Best Month: {peak_name} ({peak_month['Sunscore']:.1f})")
        print(f"Lowest Month: {low_name} ({low_month['Sunscore']:.1f})")
        print(f"Seasonal Variation: {peak_month['Sunscore'] - low_month['Sunscore']:.1f} points")
        
        # Solar potential assessment
        if annual_avg >= 70: 
            assessment = "Excellent ⭐⭐⭐⭐⭐"
            recommendation = "Outstanding location for solar installation"
        elif annual_avg >= 55: 
            assessment = "Very Good ⭐⭐⭐⭐"
            recommendation = "Great solar potential with strong returns"
        elif annual_avg >= 40: 
            assessment = "Good ⭐⭐⭐"
            recommendation = "Solid solar investment opportunity"
        elif annual_avg >= 25: 
            assessment = "Fair ⭐⭐"
            recommendation = "Moderate solar potential, consider other factors"
        else: 
            assessment = "Limited ⭐"
            recommendation = "Solar may not be optimal for this location"
        
        print(f"\n🎯 Overall Assessment: {assessment}")
        print(f"💡 Recommendation: {recommendation}")
        print(f"🔍 Data Confidence: {confidence:.0f}%")
        
        # Seasonal insights
        summer_months = sunscore_data[sunscore_data['Month'].isin([6, 7, 8])]
        winter_months = sunscore_data[sunscore_data['Month'].isin([12, 1, 2])]
        
        if len(summer_months) > 0 and len(winter_months) > 0:
            summer_avg = summer_months['Sunscore'].mean()
            winter_avg = winter_months['Sunscore'].mean()
            
            print(f"\n🌤️  Seasonal Analysis:")
            print(f"   Summer Average: {summer_avg:.1f}")
            print(f"   Winter Average: {winter_avg:.1f}")
            if winter_avg > 0:
                print(f"   Summer/Winter Ratio: {summer_avg/winter_avg:.1f}x")
        
        # Try ultra-precision method if available
        try:
            print("\n" + "="*50)
            print("ULTRA-PRECISION ANALYSIS")
            print("="*50)
            
            ultra_data, ultra_conf = calculator.get_monthly_sunscore_precision(
                user_lat, user_lon,
                user_zip=user_zip,
                k_neighbors=5,
                max_distance_km=20
            )
            
            if ultra_data is not None and len(ultra_data) > 0:
                print("\nUltra-Precision Results:")
                if 'Sunscore_uncertainty' in ultra_data.columns:
                    for _, row in ultra_data.iterrows():
                        month_name = month_names[int(row['Month'])-1]
                        print(f"{month_name}: {row['Sunscore']:.1f} ± {row['Sunscore_uncertainty']:.1f}")
                else:
                    print(ultra_data.round(1))
                
                print(f"\nUltra-Precision Confidence: {ultra_conf:.1f}%")
            else:
                print("Ultra-precision analysis not available")
                
        except Exception as e:
            print(f"Ultra-precision method not available: {e}")
            
    else:
        print("❌ No solar data available for this location.")
        print("Try expanding the search radius or check coordinates.")
        print("Ultra-precision method not available.")