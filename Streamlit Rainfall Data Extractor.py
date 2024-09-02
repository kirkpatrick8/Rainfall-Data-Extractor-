import streamlit as st
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import pandas as pd
from datetime import datetime, timedelta
import io

def extract_rainfall(uploaded_file, target_x, target_y):
    # Read the uploaded file into a BytesIO object
    file_content = io.BytesIO(uploaded_file.read())
    
    with nc.Dataset('in-memory-file', mode='r', memory=file_content.read()) as ds:
        st.write(f"Variables in the file: {list(ds.variables.keys())}")
        
        rainfall_var = 'rainfall_amount'
        if rainfall_var not in ds.variables:
            raise ValueError(f"No rainfall variable found in the uploaded file")
        
        st.write(f"Using rainfall variable: {rainfall_var}")
        
        # Read x and y coordinates
        x = ds.variables['x'][:]
        y = ds.variables['y'][:]
        
        # Find the nearest grid point
        x_idx = np.abs(x - target_x).argmin()
        y_idx = np.abs(y - target_y).argmin()
        st.write(f"Nearest grid point: x={x[x_idx]}, y={y[y_idx]}")
        
        # Extract rainfall data for the specific point
        rainfall = ds.variables[rainfall_var][:, y_idx, x_idx]
        time = ds.variables['time'][:]
        
        # Print rainfall variable attributes
        st.write(f"Rainfall variable attributes: {ds.variables[rainfall_var].ncattrs()}")
        
        # Apply scale factor and handle fill values if they exist
        scale_factor = ds.variables[rainfall_var].getncattr('scale_factor') if 'scale_factor' in ds.variables[rainfall_var].ncattrs() else 1
        fill_value = ds.variables[rainfall_var]._FillValue if hasattr(ds.variables[rainfall_var], '_FillValue') else None
        
        st.write(f"Scale factor: {scale_factor}")
        st.write(f"Fill value: {fill_value}")
        
        st.write("Raw rainfall data (first 5 values):", rainfall[:5])
        
        # Handle masked array
        if ma.is_masked(rainfall):
            rainfall = rainfall.filled(np.nan)  # Replace masked values with NaN
        
        rainfall = rainfall * scale_factor
        st.write("Processed rainfall data (first 5 values):", rainfall[:5])
        
        # Convert time to datetime
        time_units = ds.variables['time'].units
        time_calendar = ds.variables['time'].calendar if hasattr(ds.variables['time'], 'calendar') else 'standard'
        dates = nc.num2date(time, units=time_units, calendar=time_calendar)
        
        return dates, rainfall

def main():
    st.title("Rainfall Data Extractor")

    st.header("Input Parameters")
    
    # File upload
    st.subheader("NetCDF File Upload")
    uploaded_files = st.file_uploader("Choose NetCDF file(s)", type="nc", accept_multiple_files=True)

    # Coordinate system selection
    st.subheader("Coordinate System")
    coord_system = st.radio("Select the coordinate system you're using:",
                            ("Irish Grid", "British National Grid", "Latitude/Longitude"),
                            help="Choose the coordinate system that matches your input coordinates")

    # Coordinate inputs
    st.subheader("Target Coordinates")
    if coord_system == "Irish Grid" or coord_system == "British National Grid":
        col1, col2 = st.columns(2)
        with col1:
            target_x = st.number_input("Enter Easting coordinate:", value=341914.1)
        with col2:
            target_y = st.number_input("Enter Northing coordinate:", value=392651.1)
    else:  # Latitude/Longitude
        col1, col2 = st.columns(2)
        with col1:
            target_y = st.number_input("Enter Latitude:", value=54.5, min_value=-90.0, max_value=90.0)
        with col2:
            target_x = st.number_input("Enter Longitude:", value=-5.9, min_value=-180.0, max_value=180.0)

    if st.button("Extract Rainfall Data"):
        if not uploaded_files:
            st.error("Please upload at least one NetCDF file.")
            return

        all_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing file: {uploaded_file.name}")
            try:
                dates, rainfall = extract_rainfall(uploaded_file, target_x, target_y)
                all_data.extend(zip(dates, rainfall))
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))

        # Create a DataFrame and sort by date
        df = pd.DataFrame(all_data, columns=['date', 'rainfall'])
        df = df.sort_values('date')
        
        # Remove rows with NaN rainfall values
        df = df.dropna(subset=['rainfall'])

        # Display results
        st.success("Data extraction completed!")
        st.subheader("Results")
        st.write("First few rows of the dataframe:")
        st.write(df.head())
        st.write("\nData summary:")
        st.write(df.describe())

        # Provide download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='rainfall_time_series.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()