import streamlit as st
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import pandas as pd
from datetime import datetime, timedelta
import os

def extract_data(file_path, variable, target_x, target_y):
    with nc.Dataset(file_path, 'r') as ds:
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
    
    # File path input
    st.subheader("NetCDF File Path")
    file_path = st.text_input("Enter the full path to your NetCDF file or directory containing NetCDF files:")

    if file_path:
        if os.path.isdir(file_path):
            # If it's a directory, list all .nc files
            nc_files = [f for f in os.listdir(file_path) if f.endswith('.nc')]
            selected_file = st.selectbox("Select a NetCDF file", nc_files)
            full_path = os.path.join(file_path, selected_file)
        else:
            full_path = file_path

        # Get variables from the file
        with nc.Dataset(full_path, 'r') as ds:
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
            try:
                dates, data = extract_data(full_path, selected_variable, target_x, target_y)
                
                # Create a DataFrame and sort by date
                df = pd.DataFrame({
                    'date': dates,
                    selected_variable: data
                })
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
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
