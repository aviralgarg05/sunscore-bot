import pandas as pd
from geopy.distance import geodesic
import numpy as np
from sklearn.neighbors import BallTree
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class SunScoreCalculator:
    def __init__(self, file_path='solar_data.csv'):
        """Initialize with optimized data loading and indexing"""
        print("Loading and indexing solar data...")
        self.df = pd.read_csv(file_path)
        self.df.columns = self.df.columns.str.strip()
        
        # Preprocess data once
        self._preprocess_data()
        self._create_spatial_index()
        print(f"Loaded {len(self.df)} records with spatial indexing")
    
    def _preprocess_data(self):
        """Preprocess data for better performance"""
        # Convert timestamp once
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        
        # Remove invalid timestamps
        self.df = self.df.dropna(subset=['timestamp'])
        
        # Extract year and month
        self.df['Year'] = self.df['timestamp'].dt.year
        self.df['Month'] = self.df['timestamp'].dt.month
        
        # Fix zip code formatting - handle both string and numeric zip codes
        self.df['zip_code'] = self.df['zip_code'].astype(str).str.replace('.0', '', regex=False)
        self.df['zip_code'] = self.df['zip_code'].str.zfill(5)  # Pad with leading zeros
        
        # Create unique location dataframe for spatial indexing
        self.locations_df = self.df[['latitude', 'longitude', 'zip_code']].drop_duplicates().reset_index(drop=True)
        
        # Convert zip codes to numeric for proximity calculations
        self.locations_df['zip_numeric'] = pd.to_numeric(self.locations_df['zip_code'], errors='coerce')
        
        print(f"Sample zip codes in data: {sorted(self.locations_df['zip_code'].unique())[:10]}")
        
    def _create_spatial_index(self):
        """Create BallTree for fast spatial queries with haversine metric"""
        # Convert to radians for haversine distance
        coords_rad = np.radians(self.locations_df[['latitude', 'longitude']].values)
        self.kdtree = BallTree(coords_rad, metric='haversine')
        
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate haversine distance between two points"""
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _get_nearby_locations(self, user_lat, user_lon, k=5, max_distance_km=50):
        """Find k nearest locations within max_distance using BallTree"""
        user_coords_rad = np.radians([[user_lat, user_lon]])
        
        # Query BallTree for nearest neighbors
        distances_rad, indices = self.kdtree.query(user_coords_rad, k=min(k, len(self.locations_df)))
        
        # Convert back to km
        distances_km = distances_rad[0] * 6371
        indices = indices[0]
        
        # Filter by max distance
        valid_mask = distances_km <= max_distance_km
        if not np.any(valid_mask):
            # If no points within max_distance, take the closest one
            valid_mask = np.array([True] + [False] * (len(distances_km) - 1))
        
        nearby_locations = []
        for i, idx in enumerate(indices[valid_mask]):
            location = self.locations_df.iloc[idx]
            nearby_locations.append({
                'zip_code': location['zip_code'],
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'distance': distances_km[i],
                'weight': 1 / (1 + distances_km[i])  # Inverse distance weighting
            })
        
        return nearby_locations
    
    def _calculate_zip_distance(self, zip1, zip2):
        """Calculate proximity between two US zip codes using numeric distance"""
        try:
            zip1_num = int(str(zip1).zfill(5))
            zip2_num = int(str(zip2).zfill(5))
            
            # For US zip codes, numeric proximity often correlates with geographic proximity
            # Use a scaled approach where closer numbers = higher similarity
            numeric_diff = abs(zip1_num - zip2_num)
            
            # Special handling for different zip code ranges
            if numeric_diff == 0:
                return 0  # Exact match
            elif numeric_diff <= 5:
                return 1  # Very close (likely same area)
            elif numeric_diff <= 20:
                return 2  # Close (likely same city/region)
            elif numeric_diff <= 100:
                return 3  # Moderate distance (same metro area)
            elif numeric_diff <= 500:
                return 4  # Same state/region
            else:
                return 5  # Far apart
                
        except (ValueError, TypeError):
            return 10  # Invalid zip codes
    
    def _find_nearby_zip_codes(self, user_zip, max_zip_distance=3, limit=10):
        """Find zip codes that are numerically close to the user's zip code"""
        if not user_zip:
            return []
            
        user_zip_normalized = str(user_zip).zfill(5)
        
        try:
            user_zip_num = int(user_zip_normalized)
        except ValueError:
            return []
        
        # Calculate zip code distances for all locations
        zip_distances = []
        for _, location in self.locations_df.iterrows():
            zip_dist = self._calculate_zip_distance(user_zip_num, location['zip_code'])
            if zip_dist <= max_zip_distance:
                zip_distances.append({
                    'zip_code': location['zip_code'],
                    'latitude': location['latitude'],
                    'longitude': location['longitude'],
                    'zip_distance': zip_dist,
                    'zip_numeric': location['zip_numeric']
                })
        
        # Sort by zip distance (closer zip codes first)
        zip_distances.sort(key=lambda x: (x['zip_distance'], abs(user_zip_num - x['zip_numeric'])))
        
        return zip_distances[:limit]
    
    def _get_nearby_locations_with_zip(self, user_lat, user_lon, user_zip=None, k=5, max_distance_km=50):
        """Enhanced search using both zip code numeric proximity and spatial proximity"""
        nearby_locations = []
        
        # Strategy 1: If zip code is provided, find numerically close zip codes
        if user_zip:
            user_zip_normalized = str(user_zip).zfill(5)
            print(f"Searching for zip codes close to: {user_zip_normalized}")
            
            # Find nearby zip codes using numeric proximity
            nearby_zips = self._find_nearby_zip_codes(user_zip_normalized, max_zip_distance=3, limit=k*2)
            
            if nearby_zips:
                print(f"Found {len(nearby_zips)} numerically close zip codes")
                
                for zip_info in nearby_zips:
                    # Calculate actual geographic distance
                    geo_distance = self._haversine_distance(
                        user_lat, user_lon, 
                        zip_info['latitude'], zip_info['longitude']
                    )
                    
                    if geo_distance <= max_distance_km:
                        # Calculate weight based on both zip proximity and geographic distance
                        zip_weight_factor = {0: 10, 1: 8, 2: 6, 3: 4}.get(zip_info['zip_distance'], 2)
                        geo_weight_factor = 1 / (1 + geo_distance * 0.1)
                        
                        combined_weight = zip_weight_factor * geo_weight_factor
                        
                        match_type = 'exact_zip' if zip_info['zip_distance'] == 0 else 'close_zip'
                        
                        nearby_locations.append({
                            'zip_code': zip_info['zip_code'],
                            'latitude': zip_info['latitude'],
                            'longitude': zip_info['longitude'],
                            'distance': geo_distance,
                            'weight': combined_weight,
                            'match_type': match_type,
                            'zip_distance': zip_info['zip_distance']
                        })
                        
                        print(f"  Zip: {zip_info['zip_code']}, Geo Distance: {geo_distance:.2f}km, "
                              f"Zip Distance: {zip_info['zip_distance']}, Weight: {combined_weight:.2f}")
            else:
                print(f"No numerically close zip codes found for: {user_zip_normalized}")
        
        # Strategy 2: Fill remaining slots with spatial search if needed
        current_count = len(nearby_locations)
        if current_count < k:
            remaining_needed = k - current_count
            
            user_coords_rad = np.radians([[user_lat, user_lon]])
            
            # Search for more locations spatially
            spatial_k = max(remaining_needed * 2, 10)
            distances_rad, indices = self.kdtree.query(
                user_coords_rad, 
                k=min(spatial_k, len(self.locations_df))
            )
            
            distances_km = distances_rad[0] * 6371
            indices = indices[0]
            
            # Add spatial neighbors (excluding already added locations)
            existing_zips = {loc['zip_code'] for loc in nearby_locations}
            
            for i, idx in enumerate(indices):
                if len(nearby_locations) >= k:
                    break
                    
                if distances_km[i] > max_distance_km:
                    continue
                    
                location = self.locations_df.iloc[idx]
                
                if location['zip_code'] not in existing_zips:
                    nearby_locations.append({
                        'zip_code': location['zip_code'],
                        'latitude': location['latitude'],
                        'longitude': location['longitude'],
                        'distance': distances_km[i],
                        'weight': 1 / (1 + distances_km[i] * 0.1),
                        'match_type': 'spatial',
                        'zip_distance': None
                    })
        
        # Sort by weight (highest first) and limit to k locations
        nearby_locations.sort(key=lambda x: x['weight'], reverse=True)
        
        return nearby_locations[:k]
    
    def _calculate_weighted_sunscore(self, nearby_locations):
        """Calculate weighted sunscore from multiple nearby locations"""
        all_monthly_data = []
        total_weight = sum(loc['weight'] for loc in nearby_locations)
        
        for location in nearby_locations:
            zip_code = location['zip_code']
            weight = location['weight'] / total_weight
            
            # Get data for this zip code
            zip_data = self.df[self.df['zip_code'] == zip_code].copy()
            
            if zip_data.empty:
                continue
                
            # Calculate monthly averages
            monthly_avg = zip_data.groupby(['Year', 'Month']).agg({
                'ghi': 'mean',
                'dni': 'mean',
                'dhi': 'mean'
            }).reset_index()
            
            # Enhanced sunscore calculation with seasonal adjustments
            monthly_avg['raw_sunscore'] = self._calculate_enhanced_sunscore(monthly_avg)
            monthly_avg['weight'] = weight
            monthly_avg['zip_code'] = zip_code
            monthly_avg['distance'] = location['distance']
            
            all_monthly_data.append(monthly_avg)
        
        if not all_monthly_data:
            return None, 0
            
        # Combine all data
        combined_df = pd.concat(all_monthly_data, ignore_index=True)
        
        # Calculate weighted average sunscore by month
        weighted_monthly = combined_df.groupby(['Year', 'Month']).apply(
            lambda x: pd.Series({
                'Sunscore': np.average(x['raw_sunscore'], weights=x['weight']),
                'data_points': len(x),
                'avg_distance': np.average(x['distance'], weights=x['weight'])
            })
        ).reset_index()
        
        # Normalize to 0-100 scale
        if len(weighted_monthly) > 0:
            sunscore_min = weighted_monthly['Sunscore'].min()
            sunscore_max = weighted_monthly['Sunscore'].max()
            if sunscore_max > sunscore_min:
                weighted_monthly['Sunscore'] = 100 * (weighted_monthly['Sunscore'] - sunscore_min) / (sunscore_max - sunscore_min)
            else:
                weighted_monthly['Sunscore'] = 50  # Default if no variation
        
        # Calculate enhanced confidence
        confidence = self._calculate_enhanced_confidence(weighted_monthly, nearby_locations)
        
        return weighted_monthly[['Year', 'Month', 'Sunscore']], confidence
    
    def _calculate_enhanced_sunscore(self, monthly_avg):
        """Enhanced sunscore calculation with seasonal adjustments"""
        # Improved weights based on solar energy research
        alpha, beta, gamma = 0.6, 0.25, 0.15  # GHI is most important for overall solar potential
        
        # Base sunscore
        base_score = (alpha * monthly_avg['ghi'] + 
                     beta * monthly_avg['dni'] + 
                     gamma * monthly_avg['dhi'])
        
        # Seasonal adjustment factor (boost summer months, reduce winter months)
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * (monthly_avg['Month'] - 3) / 12)
        
        return base_score * seasonal_factor
    
    def _calculate_enhanced_confidence(self, monthly_data, nearby_locations):
        """Calculate enhanced confidence based on multiple factors"""
        if monthly_data is None or len(monthly_data) == 0:
            return 0
        
        # Factor 1: Average distance to data points (closer = higher confidence)
        avg_distance = np.mean([loc['distance'] for loc in nearby_locations])
        distance_confidence = max(0, 100 - (avg_distance * 3))  # 3% reduction per km
        
        # Factor 2: Number of nearby locations used
        location_confidence = min(100, len(nearby_locations) * 25)  # Up to 4 locations for 100%
        
        # Factor 3: Data coverage (how many months we have data for)
        months_covered = len(monthly_data)
        coverage_confidence = min(100, (months_covered / 12) * 100)
        
        # Factor 4: Data consistency (lower std = higher confidence)
        if len(monthly_data) > 1:
            sunscore_std = monthly_data['Sunscore'].std()
            consistency_confidence = max(0, 100 - sunscore_std)
        else:
            consistency_confidence = 50
        
        # Factor 5: Data recency and density
        avg_data_points = monthly_data['data_points'].mean()
        density_confidence = min(100, avg_data_points * 10)  # More data points = higher confidence
        
        # Weighted combination of all factors
        confidence = (distance_confidence * 0.3 + 
                     location_confidence * 0.2 + 
                     coverage_confidence * 0.2 + 
                     consistency_confidence * 0.2 + 
                     density_confidence * 0.1)
        
        return min(100, confidence)
    
    def _calculate_enhanced_confidence_with_zip(self, monthly_data, nearby_locations, user_zip=None):
        """Enhanced confidence calculation considering zip code numeric proximity"""
        if monthly_data is None or len(monthly_data) == 0:
            return 0
        
        # Factor 1: Zip code proximity bonus (much more accurate now)
        zip_confidence = 30  # Base confidence
        if user_zip:
            user_zip_normalized = str(user_zip).zfill(5)
            
            # Count exact and close matches
            exact_matches = sum(1 for loc in nearby_locations 
                              if loc.get('zip_distance') == 0)
            very_close_matches = sum(1 for loc in nearby_locations 
                                   if loc.get('zip_distance') == 1)
            close_matches = sum(1 for loc in nearby_locations 
                              if loc.get('zip_distance') in [2, 3])
            
            if exact_matches > 0:
                zip_confidence = min(100, 95 + exact_matches * 2)
                print(f"Exact zip matches found: {exact_matches} - Very high confidence")
            elif very_close_matches > 0:
                zip_confidence = min(100, 85 + very_close_matches * 3)
                print(f"Very close zip matches found: {very_close_matches} - High confidence")
            elif close_matches > 0:
                zip_confidence = min(100, 70 + close_matches * 2)
                print(f"Close zip matches found: {close_matches} - Good confidence")
            else:
                zip_confidence = 40
                print("No close zip codes found - using spatial data only")
        
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
        avg_data_points = monthly_data['data_points'].mean()
        density_confidence = min(100, avg_data_points * 10)
        
        # Weighted combination with higher weight on zip matching
        confidence = (zip_confidence * 0.4 + 
                     distance_confidence * 0.2 + 
                     location_confidence * 0.15 + 
                     coverage_confidence * 0.1 + 
                     consistency_confidence * 0.1 + 
                     density_confidence * 0.05)
        
        return min(100, confidence)
    
    def get_monthly_sunscore(self, user_lat, user_lon, user_zip=None, k_neighbors=5, max_distance_km=50):
        """Main method to get monthly sunscore with enhanced zip-aware accuracy"""
        print(f"Calculating sunscore for coordinates: ({user_lat}, {user_lon})")
        if user_zip:
            print(f"Using zip code: {user_zip} for enhanced search")
        
        # Find nearby locations using enhanced zip-aware search
        nearby_locations = self._get_nearby_locations_with_zip(
            user_lat, user_lon, user_zip, k_neighbors, max_distance_km
        )
        
        if not nearby_locations:
            return "No data found within reasonable distance.", 0
        
        print(f"Using {len(nearby_locations)} nearby locations for interpolation")
        for i, loc in enumerate(nearby_locations[:3]):
            print(f"  {i+1}. Zip: {loc['zip_code']}, Distance: {loc['distance']:.2f}km, Type: {loc['match_type']}")
        
        # Calculate weighted sunscore using existing method
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
            
            monthly_avg['raw_sunscore'] = self._calculate_enhanced_sunscore(monthly_avg)
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
        
        if len(weighted_monthly) > 0:
            sunscore_min = weighted_monthly['Sunscore'].min()
            sunscore_max = weighted_monthly['Sunscore'].max()
            if sunscore_max > sunscore_min:
                weighted_monthly['Sunscore'] = 100 * (weighted_monthly['Sunscore'] - sunscore_min) / (sunscore_max - sunscore_min)
            else:
                weighted_monthly['Sunscore'] = 50
        
        # Use enhanced confidence calculation with zip awareness
        confidence = self._calculate_enhanced_confidence_with_zip(
            weighted_monthly, nearby_locations, user_zip
        )
        
        return weighted_monthly[['Year', 'Month', 'Sunscore']], confidence

# Initialize calculator
calculator = SunScoreCalculator('solar_data.csv')

# Example usage with your coordinates
user_lat = 42.04783608530275
user_lon = -72.62718033091582

# Get enhanced sunscore with multiple nearby points
sunscore_data, confidence_level = calculator.get_monthly_sunscore(
    user_lat, user_lon, 
    k_neighbors=5,  # Use 5 nearest locations
    max_distance_km=50  # Within 50km radius
)

# Display results
print("\nEnhanced Monthly Sunscore Data (Scaled to 100):")
print(sunscore_data)
print(f"\nEnhanced Confidence Level: {confidence_level:.2f}%")

# Additional insights
if hasattr(sunscore_data, '__len__') and len(sunscore_data) > 0:
    annual_avg = sunscore_data['Sunscore'].mean()
    peak_month = sunscore_data.loc[sunscore_data['Sunscore'].idxmax()]
    low_month = sunscore_data.loc[sunscore_data['Sunscore'].idxmin()]
    
    print(f"\nAdditional Insights:")
    print(f"Annual Average Sunscore: {annual_avg:.1f}")
    print(f"Peak Month: {peak_month['Month']}/{peak_month['Year']} (Score: {peak_month['Sunscore']:.1f})")
    print(f"Lowest Month: {low_month['Month']}/{low_month['Year']} (Score: {low_month['Sunscore']:.1f})")

# Example usage with zip code (now with proper numeric proximity)
user_lat = 42.04783608530275
user_lon = -72.62718033091582
user_zip = "01013"  # This should now find closer zip codes like 01012, 01014, etc.

# Get enhanced sunscore with zip code information
sunscore_data, confidence_level = calculator.get_monthly_sunscore(
    user_lat, user_lon, 
    user_zip=user_zip,  # Include zip code for better accuracy
    k_neighbors=5,
    max_distance_km=50
)

# Display results
print("\nEnhanced Monthly Sunscore Data (Scaled to 100) with Improved Zip Code Matching:")
print(sunscore_data)
print(f"\nEnhanced Confidence Level: {confidence_level:.2f}%")

# Additional insights
if hasattr(sunscore_data, '__len__') and len(sunscore_data) > 0:
    annual_avg = sunscore_data['Sunscore'].mean()
    peak_month = sunscore_data.loc[sunscore_data['Sunscore'].idxmax()]
    low_month = sunscore_data.loc[sunscore_data['Sunscore'].idxmin()]
    
    print(f"\nAdditional Insights with Improved Zip Code Matching:")
    print(f"Annual Average Sunscore: {annual_avg:.1f}")
    print(f"Peak Month: {peak_month['Month']}/{peak_month['Year']} (Score: {peak_month['Sunscore']:.1f})")
    print(f"Lowest Month: {low_month['Month']}/{low_month['Year']} (Score: {low_month['Sunscore']:.1f})")
