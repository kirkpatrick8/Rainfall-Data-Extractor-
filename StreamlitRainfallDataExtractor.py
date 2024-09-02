import streamlit as st
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import pandas as pd
from datetime import datetime, timedelta
import io

def extract_data(uploaded_file, variable, target_x, target_y):
    file_content = io.BytesIO(uploaded_file.read())
    
    with nc.Dataset('in-memory-file', mode='r', memory=file_content.read()) as ds:
        st.write(f"Variables in the file: {list(ds.variables.keys())}")
        
        if variable not in ds.variables:
            raise ValueError(f"Selected variable '{variable}' not found in the file")
        
        st.write(f"Using variable: {variable}")
        
        # Read x and y coordinates
        x = ds.variables['x'][:]
        y = ds.variables['y'][:]
        
        # Find the nearest grid point
        x_idx = np.abs(x - target_x).argmin()
        y_idx = np.abs(y - target_y).argmin()
        st.write(f"Nearest grid point: x={x[x_idx]}, y={y[y_idx]}")
        
        # Extract data for the specific point
        data = ds.variables[variable][:, y_idx, x_idx]
        time = ds.variables['time'][:]
        
        # Print variable attributes
        st.write(f"Variable attributes: {ds.variables[variable].ncattrs()}")
        
        # Apply scale factor and handle fill values if they exist
        scale_factor = ds.variables[variable].getncattr('scale_factor') if 'scale_factor' in ds.variables[variable].ncattrs() else 1
        fill_value = ds.variables[variable]._FillValue if hasattr(ds.variables[variable], '_FillValue') else None
        
        st.write(f"Scale factor: {scale_factor}")
        st.write(f"Fill value: {fill_value}")
        
        st.write(f"Raw {variable} data (first 5 values):", data[:5])
        
        # Handle masked array
        if ma.is_masked(data):
            data = data.filled(np.nan)  # Replace masked values with NaN
        
        data = data * scale_factor
        st.write(f"Processed {variable} data (first 5 values):", data[:5])
        
        # Convert time to datetime
        time_units = ds.variables['time'].units
        time_calendar = ds.variables['time'].calendar if hasattr(ds.variables['time'], 'calendar') else 'standard'
        dates = nc.num2date(time, units=time_units, calendar=time_calendar)
        
        return dates, data

def main():
    st.title("NetCDF Data Extractor")

    st.header("Input Parameters")
    
    # File upload
    st.subheader("NetCDF File Upload")
    uploaded_files = st.file_uploader("Choose NetCDF file(s)", type="nc", accept_multiple_files=True)

    if uploaded_files:
        # Get variables from the first file
        with nc.Dataset('in-memory-file', mode='r', memory=io.BytesIO(uploaded_files[0].read()).read()) as ds:
            variables = list(ds.variables.keys())
        
        # Variable selection
        selected_variable = st.selectbox("Select a variable to extract", variables)

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

    if st.button("Extract Data"):
        if not uploaded_files:
            st.error("Please upload at least one NetCDF file.")
            return

        all_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing file: {uploaded_file.name}")
            try:
                dates, data = extract_data(uploaded_file, selected_variable, target_x, target_y)
                all_data.extend(zip(dates, data))
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))

        # Create a DataFrame and sort by date
        df = pd.DataFrame(all_data, columns=['date', selected_variable])
        df = df.sort_values('date')
        
        # Remove rows with NaN values
        df = df.dropna()

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
            file_name=f'{selected_variable}_time_series.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()
