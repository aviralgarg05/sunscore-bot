import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import warnings

# Try importing geopy with better error handling
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    GEOPY_AVAILABLE = True
except ImportError as e:
    GEOPY_AVAILABLE = False
    st.error(f"⚠️ Geopy not available: {e}")
    st.info("Please install geopy: `pip install geopy`")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    st.warning("📡 Requests module not available. Some features may be limited.")

# Import the calculator with error handling
try:
    from app import SunScoreCalculator
    CALCULATOR_AVAILABLE = True
except ImportError as e:
    CALCULATOR_AVAILABLE = False
    st.error(f"❌ Cannot import SunScoreCalculator: {e}")
    st.stop()

# Optional imports with fallbacks
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import geocoder
    GEOCODER_AVAILABLE = True
except ImportError:
    GEOCODER_AVAILABLE = False

try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except ImportError:
    PYDECK_AVAILABLE = False

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🌞 SunScore Analytics",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_calculator():
    """Load the SunScore calculator with caching"""
    try:
        return SunScoreCalculator('solar_data.csv')
    except Exception as e:
        st.error(f"Failed to load solar data: {e}")
        return None

class GeocodingService:
    """Enhanced geocoding service with multiple fallback options"""
    
    def __init__(self):
        self.services = []
        
        if GEOPY_AVAILABLE:
            try:
                self.nominatim = Nominatim(user_agent="sunscore_app_v1.0")
                self.services.append('nominatim')
            except Exception as e:
                st.warning(f"Failed to initialize Nominatim: {e}")
        
        if GEOCODER_AVAILABLE:
            self.services.extend(['arcgis', 'bing'])
        
        if not self.services:
            st.error("❌ No geocoding services available. Please install geopy: `pip install geopy`")
    
    def geocode_address(self, address):
        """Geocode address using multiple services with fallbacks"""
        if not GEOPY_AVAILABLE:
            st.error("Geocoding not available. Please install geopy.")
            return []
        
        results = []
        
        # Method 1: Nominatim (OpenStreetMap)
        try:
            location = self.nominatim.geocode(address, timeout=10)
            if location:
                results.append({
                    'service': 'Nominatim (OpenStreetMap)',
                    'latitude': location.latitude,
                    'longitude': location.longitude,
                    'formatted_address': location.address,
                    'confidence': 0.9
                })
        except Exception as e:
            st.warning(f"Nominatim geocoding failed: {e}")
        
        # Method 2: ArcGIS (backup) - only if geocoder is available
        if GEOCODER_AVAILABLE and not results:
            try:
                import geocoder
                g = geocoder.arcgis(address)
                if g.ok:
                    results.append({
                        'service': 'ArcGIS',
                        'latitude': g.latlng[0],
                        'longitude': g.latlng[1],
                        'formatted_address': g.address,
                        'confidence': g.confidence if hasattr(g, 'confidence') else 0.8
                    })
            except Exception as e:
                st.warning(f"ArcGIS geocoding failed: {e}")
        
        return results
    
    def geocode_zipcode(self, zipcode):
        """Specialized ZIP code geocoding"""
        if not GEOPY_AVAILABLE:
            st.error("ZIP code geocoding not available. Please install geopy.")
            return None
        
        # Clean ZIP code
        zipcode = str(zipcode).strip().zfill(5)
        
        # Try with ZIP code specific formatting
        queries = [
            zipcode,
            f"{zipcode}, USA",
            f"ZIP {zipcode}, United States"
        ]
        
        for query in queries:
            try:
                location = self.nominatim.geocode(query, timeout=10)
                if location:
                    return {
                        'service': 'ZIP Code Lookup',
                        'latitude': location.latitude,
                        'longitude': location.longitude,
                        'formatted_address': location.address,
                        'confidence': 0.95,
                        'zipcode': zipcode
                    }
            except Exception:
                continue
        
        return None

def create_solar_gauge(score, title="Solar Score"):
    """Create an animated gauge chart for solar score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24}},
        delta = {'reference': 50, 'increasing': {'color': "RebeccaPurple"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': 'lightgray'},
                {'range': [25, 50], 'color': 'yellow'},
                {'range': [50, 75], 'color': 'orange'},
                {'range': [75, 100], 'color': 'green'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"},
        height=300
    )
    
    return fig

def create_monthly_chart(sunscore_data):
    """Create an interactive monthly solar score chart"""
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Prepare data
    df = sunscore_data.copy()
    df['Month_Name'] = df['Month'].apply(lambda x: month_names[int(x)-1])
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Main line chart
    fig.add_trace(
        go.Scatter(
            x=df['Month_Name'],
            y=df['Sunscore'],
            mode='lines+markers',
            name='Solar Score',
            line=dict(color='gold', width=4),
            marker=dict(size=10, color='orange'),
            hovertemplate='<b>%{x}</b><br>Solar Score: %{y:.1f}<extra></extra>'
        )
    )
    
    # Add area fill
    fig.add_trace(
        go.Scatter(
            x=df['Month_Name'],
            y=df['Sunscore'],
            fill='tonexty',
            mode='none',
            name='Solar Potential Area',
            fillcolor='rgba(255, 165, 0, 0.3)',
            showlegend=False
        )
    )
    
    # Add average line
    avg_score = df['Sunscore'].mean()
    fig.add_hline(
        y=avg_score,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Annual Average: {avg_score:.1f}",
        annotation_position="top right"
    )
    
    # Styling
    fig.update_layout(
        title={
            'text': '🌞 Monthly Solar Potential',
            'x': 0.5,
            'font': {'size': 24, 'color': 'darkblue'}
        },
        xaxis_title="Month",
        yaxis_title="Solar Score (0-100)",
        template="plotly_white",
        hovermode='x unified',
        height=500
    )
    
    # Add custom styling
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', range=[0, 100])
    
    return fig

def create_seasonal_comparison(sunscore_data):
    """Create seasonal comparison charts"""
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    df = sunscore_data.copy()
    df['Month_Name'] = df['Month'].apply(lambda x: month_names[int(x)-1])
    
    # Define seasons
    seasons = {
        'Winter': [12, 1, 2],
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Fall': [9, 10, 11]
    }
    
    seasonal_data = []
    for season, months in seasons.items():
        season_data = df[df['Month'].isin(months)]
        if len(season_data) > 0:
            seasonal_data.append({
                'Season': season,
                'Average_Score': season_data['Sunscore'].mean(),
                'Max_Score': season_data['Sunscore'].max(),
                'Min_Score': season_data['Sunscore'].min()
            })
    
    if seasonal_data:
        seasonal_df = pd.DataFrame(seasonal_data)
        
        # Create polar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=seasonal_df['Average_Score'],
            theta=seasonal_df['Season'],
            fill='toself',
            name='Seasonal Solar Potential',
            line_color='gold',
            fillcolor='rgba(255, 165, 0, 0.6)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Seasonal Solar Distribution",
            height=400
        )
        
        return fig, seasonal_df
    
    return None, None

def create_heatmap_chart(sunscore_data):
    """Create a heatmap of solar scores by year and month"""
    df = sunscore_data.copy()
    df['Month'] = df['Month'].astype(int)
    df['Year'] = df['Year'].astype(int)
    pivot = df.pivot(index='Year', columns='Month', values='Sunscore')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot.columns = [month_names[m-1] for m in pivot.columns]
    fig = px.imshow(
        pivot,
        color_continuous_scale='YlOrRd',
        aspect='auto',
        labels=dict(x="Month", y="Year", color="Solar Score"),
        title="Solar Score Heatmap"
    )
    fig.update_layout(height=350)
    return fig

def create_annual_histogram(sunscore_data):
    """Create a histogram of annual average solar scores"""
    df = sunscore_data.copy()
    annual = df.groupby('Year')['Sunscore'].mean().reset_index()
    fig = px.histogram(
        annual,
        x='Sunscore',
        nbins=10,
        color_discrete_sequence=['orange'],
        title="Annual Solar Score Distribution"
    )
    fig.update_layout(xaxis_title="Annual Solar Score", yaxis_title="Count", height=350)
    return fig

def colored_metric(label, value, suffix="", color="default", icon=""):
    """Display a colored metric with optional icon"""
    color_map = {
        "excellent": "#2ecc40",
        "good": "#ffdc00",
        "fair": "#ff851b",
        "limited": "#ff4136",
        "default": "#0074d9"
    }
    style = f"color:{color_map.get(color, color_map['default'])};font-size:1.3em;font-weight:bold;"
    icon_html = f"{icon} " if icon else ""
    st.markdown(f"<span style='{style}'>{icon_html}{label}: {value}{suffix}</span>", unsafe_allow_html=True)

# Main application logic
def main():
    # --- Session State Initialization ---
    if 'calculator' not in st.session_state:
        st.session_state.calculator = load_calculator()
    if 'geocoding_service' not in st.session_state:
        st.session_state.geocoding_service = GeocodingService()

    st.title("SunScore Analytics")
    st.markdown("#### Discover Your Solar Potential")

    st.sidebar.header("Location Input")
    input_options = []
    if GEOPY_AVAILABLE:
        input_options = ["Address/City", "ZIP Code", "Coordinates"]
    else:
        input_options = ["Coordinates"]

    input_method = st.sidebar.radio("Choose input method:", input_options)
    latitude, longitude, formatted_address = None, None, None

    location_changed = False

    # Remove location buttons, geocode as soon as input is provided
    if input_method == "Address/City" and GEOPY_AVAILABLE:
        address = st.sidebar.text_input("Enter address or city:")
        if address:
            results = st.session_state.geocoding_service.geocode_address(address)
            if results:
                result = results[0]
                latitude = result['latitude']
                longitude = result['longitude']
                formatted_address = result['formatted_address']
                st.session_state.selected_location = result
                location_changed = True
            else:
                st.sidebar.error("Location not found.")
    elif input_method == "ZIP Code" and GEOPY_AVAILABLE:
        zipcode = st.sidebar.text_input("Enter ZIP Code:", max_chars=5)
        if zipcode:
            result = st.session_state.geocoding_service.geocode_zipcode(zipcode)
            if result:
                latitude = result['latitude']
                longitude = result['longitude']
                formatted_address = result['formatted_address']
                st.session_state.selected_location = result
                location_changed = True
            else:
                st.sidebar.error("ZIP code not found.")
    else:
        latitude = st.sidebar.number_input("Latitude:", min_value=-90.0, max_value=90.0, value=42.0478, format="%.6f")
        longitude = st.sidebar.number_input("Longitude:", min_value=-180.0, max_value=180.0, value=-72.6272, format="%.6f")
        formatted_address = f"Coordinates: {latitude:.4f}, {longitude:.4f}"
        prev_lat = st.session_state.get('prev_latitude', None)
        prev_lon = st.session_state.get('prev_longitude', None)
        if prev_lat != latitude or prev_lon != longitude:
            location_changed = True
        st.session_state.prev_latitude = latitude
        st.session_state.prev_longitude = longitude  # fix typo

    # Use session state location if available
    if 'selected_location' in st.session_state and latitude is None:
        loc = st.session_state.selected_location
        latitude = loc['latitude']
        longitude = loc['longitude']
        formatted_address = loc['formatted_address']

    st.sidebar.header("Analysis Settings")
    k_neighbors = st.sidebar.slider("Nearby locations:", 1, 10, 3)
    max_distance = st.sidebar.slider("Search radius (km):", 5, 100, 25)
    analysis_button = st.sidebar.button("Analyze Solar Potential")

    # Only clear previous analysis if location or settings changed
    if location_changed or st.session_state.get('prev_k_neighbors', None) != k_neighbors or st.session_state.get('prev_max_distance', None) != max_distance:
        st.session_state.pop('analysis_result', None)
        st.session_state.prev_k_neighbors = k_neighbors
        st.session_state.prev_max_distance = max_distance

    # Main content area
    if latitude is not None and longitude is not None:
        st.info(f"Selected Location: {formatted_address}")

        # Only run analysis when button is pressed
        if analysis_button:
            try:
                user_zip = None
                if 'selected_location' in st.session_state and 'zipcode' in st.session_state.selected_location:
                    user_zip = st.session_state.selected_location['zipcode']

                sunscore_data, confidence = st.session_state.calculator.get_monthly_sunscore_precise(
                    latitude, longitude,
                    user_zip=user_zip,
                    k_neighbors=k_neighbors,
                    max_distance_km=max_distance
                )
                st.session_state.analysis_result = {
                    'sunscore_data': sunscore_data,
                    'confidence': confidence,
                    'latitude': latitude,
                    'longitude': longitude,
                    'formatted_address': formatted_address,
                    'user_zip': user_zip,
                    'k_neighbors': k_neighbors,
                    'max_distance': max_distance
                }
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.error("Please try again with different parameters or contact support.")

        # Display analysis result if available
        if 'analysis_result' in st.session_state:
            result = st.session_state.analysis_result
            sunscore_data = result['sunscore_data']
            confidence = result['confidence']
            latitude = result['latitude']
            longitude = result['longitude']
            formatted_address = result['formatted_address']
            user_zip = result['user_zip']
            k_neighbors = result['k_neighbors']
            max_distance = result['max_distance']

            if sunscore_data is not None and len(sunscore_data) > 0:
                annual_avg = sunscore_data['Sunscore'].mean()
                peak_month_idx = sunscore_data['Sunscore'].idxmax()
                peak_month = sunscore_data.loc[peak_month_idx]
                low_month_idx = sunscore_data['Sunscore'].idxmin()
                low_month = sunscore_data.loc[low_month_idx]
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

                st.subheader("Solar Analysis Results")
                # Improved metrics display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    colored_metric("Annual Avg", f"{annual_avg:.1f}", "/100", "excellent" if annual_avg >= 70 else "good" if annual_avg >= 55 else "fair" if annual_avg >= 40 else "limited", "🌞")
                with col2:
                    colored_metric("Peak Month", f"{month_names[int(peak_month['Month'])-1]}", f" ({peak_month['Sunscore']:.1f})", "excellent", "📈")
                with col3:
                    colored_metric("Lowest Month", f"{month_names[int(low_month['Month'])-1]}", f" ({low_month['Sunscore']:.1f})", "limited", "📉")
                with col4:
                    colored_metric("Confidence", f"{confidence:.0f}", "%", "default", "🔍")

                # Assessment
                if annual_avg >= 70:
                    assessment = "Excellent"
                    recommendation = "Outstanding location for solar installation."
                    assessment_color = "excellent"
                elif annual_avg >= 55:
                    assessment = "Very Good"
                    recommendation = "Great solar potential with strong returns."
                    assessment_color = "good"
                elif annual_avg >= 40:
                    assessment = "Good"
                    recommendation = "Solid solar investment opportunity."
                    assessment_color = "fair"
                elif annual_avg >= 25:
                    assessment = "Fair"
                    recommendation = "Moderate solar potential, consider other factors."
                    assessment_color = "fair"
                else:
                    assessment = "Limited"
                    recommendation = "Solar may not be optimal for this location."
                    assessment_color = "limited"

                colored_metric("Assessment", assessment, "", assessment_color, "⭐")
                st.markdown(f"<span style='font-size:1.1em;'>{recommendation}</span>", unsafe_allow_html=True)

                # Visualizations
                st.subheader("Visualizations")
                st.plotly_chart(create_solar_gauge(annual_avg, "Annual Solar Score"), use_container_width=True)
                st.plotly_chart(create_monthly_chart(sunscore_data), use_container_width=True)
                # New visualizations
                st.plotly_chart(create_heatmap_chart(sunscore_data), use_container_width=True)
                st.plotly_chart(create_annual_histogram(sunscore_data), use_container_width=True)

                seasonal_fig, seasonal_df = create_seasonal_comparison(sunscore_data)
                if seasonal_fig:
                    st.plotly_chart(seasonal_fig, use_container_width=True)
                    if seasonal_df is not None:
                        st.write("Seasonal Breakdown:")
                        for _, row in seasonal_df.iterrows():
                            colored_metric(row['Season'], f"{row['Average_Score']:.1f}", f" (Range: {row['Min_Score']:.1f}-{row['Max_Score']:.1f})", "default")

                st.subheader("Location & Data Points")
                
                def create_location_map(latitude, longitude, nearby_locs):
                    """Create a map visualization of the selected location and nearby data points."""
                    if FOLIUM_AVAILABLE:
                        import folium
                        m = folium.Map(location=[latitude, longitude], zoom_start=10)

                        folium.Marker([latitude, longitude], popup="Selected Location", icon=folium.Icon(color="red")).add_to(m)
                        for loc in nearby_locs:
                            folium.CircleMarker(
                                location=[loc['latitude'], loc['longitude']],
                                radius=5,
                                color="blue",
                                fill=True,
                                fill_color="blue",
                                popup=f"Score: {loc.get('Sunscore', 'N/A')}"
                            ).add_to(m)
                        return m
                    elif PYDECK_AVAILABLE:
                        import pydeck as pdk
                        data = [{
                            "lat": latitude,
                            "lon": longitude,
                            "type": "Selected"
                        }] + [
                            {"lat": loc['latitude'], "lon": loc['longitude'], "type": "Nearby"} for loc in nearby_locs
                        ]
                        layer = pdk.Layer(
                            "ScatterplotLayer",
                            data,
                            get_position='[lon, lat]',
                            get_color='[255, 0, 0]' if data[0]["type"] == "Selected" else '[0, 0, 255]',
                            get_radius=100,
                        )
                        view_state = pdk.ViewState(latitude=latitude, longitude=longitude, zoom=10)
                        return pdk.Deck(layers=[layer], initial_view_state=view_state)
                    else:
                        # Fallback to Plotly
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        fig.add_trace(go.Scattermapbox(
                            lat=[latitude],
                            lon=[longitude],
                            mode='markers',
                            marker=dict(size=14, color='red'),
                            name='Selected Location'
                        ))
                        if nearby_locs:
                            fig.add_trace(go.Scattermapbox(
                                lat=[loc['latitude'] for loc in nearby_locs],
                                lon=[loc['longitude'] for loc in nearby_locs],
                                mode='markers',
                                marker=dict(size=8, color='blue'),
                                name='Nearby Locations'
                            ))
                        fig.update_layout(
                            mapbox_style="open-street-map",
                            mapbox_center={"lat": latitude, "lon": longitude},
                            mapbox_zoom=10,
                            margin={"r":0,"t":0,"l":0,"b":0},
                            height=400
                        )
                        return fig

                    try:
                        nearby_locs = st.session_state.calculator._get_nearby_locations_optimized(
                            latitude, longitude, user_zip, k=10, max_distance_km=max_distance
                        )
                        location_map = create_location_map(latitude, longitude, nearby_locs)
                        if FOLIUM_AVAILABLE:
                            st_folium(location_map, width=500, height=400)
                        elif PYDECK_AVAILABLE:
                            st.pydeck_chart(location_map)
                        else:
                            st.plotly_chart(location_map, use_container_width=True)
                        st.write(f"Using {len(nearby_locs)} nearby data points within {max_distance}km radius")
                    except Exception as e:
                        st.write(f"Location: {latitude:.4f}, {longitude:.4f}")
                        st.write(f"Search Radius: {max_distance}km")

                # Export options
                st.subheader("Export Results")
                csv = sunscore_data.to_csv(index=False)
                st.download_button(
                    label="Download Solar Data (CSV)",
                    data=csv,
                    file_name=f"solar_analysis_{latitude:.4f}_{longitude:.4f}.csv",
                    mime="text/csv"
                )
                summary_report = f"""
Solar Analysis Report

Location: {formatted_address}
Coordinates: {latitude:.6f}, {longitude:.6f}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Key Findings
- Annual Average Solar Score: {annual_avg:.1f}/100
- Best Month: {month_names[int(peak_month['Month'])-1]} ({peak_month['Sunscore']:.1f})
- Lowest Month: {month_names[int(low_month['Month'])-1]} ({low_month['Sunscore']:.1f})
- Assessment: {assessment}
- Data Confidence: {confidence:.1f}%

Recommendation
{recommendation}

Generated by SunScore Analytics
                """
                st.download_button(
                    label="Download Report (TXT)",
                    data=summary_report,
                    file_name=f"solar_report_{latitude:.4f}_{longitude:.4f}.txt",
                    mime="text/plain"
                )
            else:
                st.error("No solar data available for this location. Try expanding the search radius or checking a different location.")
    else:
        st.markdown("Enter your location in the sidebar to begin.")

if __name__ == "__main__":
    main()
