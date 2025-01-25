
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import boto3
from io import BytesIO

from time import time_ns, sleep
import sys
import csv
import os

from colorama import init, Fore, Style
import streamlit as st
import re
import bcrypt
import sqlite3

# Connect to SQLite database (or create it)
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create users table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT, email TEXT)''')
conn.commit()

# Function to hash passwords
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Function to check passwords
def check_password(hashed_password, user_password):
    return bcrypt.checkpw(user_password.encode(), hashed_password.encode())

init(autoreset=True)  # Initialize colorama to auto-reset colors after each print statement

# Streamlit app header and instructions
st.markdown("<h1 style = 'text-align: center; color: #001e69;'>Welcome to the LETREP25 Project!</h1>", unsafe_allow_html=True)
st.subheader("Please login with your username and password below")
st.markdown("")

st.title("Login Page")


# Registration form
if 'register' not in st.session_state:
    st.session_state['register'] = False

if st.button('Register'):
    st.session_state['register'] = True

if st.session_state['register']:
    new_username = st.text_input('New Username')
    new_password = st.text_input('New Password', type='password')
    email = st.text_input('Email')
    if st.button('Submit Registration'):
        hashed_password = hash_password(new_password)
        c.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)', (new_username, hashed_password, email))
        conn.commit()
        st.success('User registered successfully')
# Button to display all users
#if st.button('Show All Users'):
    #c.execute('SELECT * FROM users')
    #users = c.fetchall()
    #for user in users:
        #st.write(user)

# Login form
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')

if st.button('Login'):
    c.execute('SELECT password, email FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    if result and check_password(result[0], password):
        st.success(f'Welcome {username}')
        st.session_state['logged_in'] = True
        st.session_state['username'] = username
        st.session_state['email'] = result[1]
    else:
        st.error('Username/password is incorrect')

# Forgot Password form
if 'forgot_password' not in st.session_state:
    st.session_state['forgot_password'] = False

if st.button('Forgot Password'):
    st.session_state['forgot_password'] = True

if st.session_state['forgot_password']:
    reset_username = st.text_input('Enter your username to reset password')
    new_password = st.text_input('Enter new password', type='password')
    if st.button('Submit New Password'):
        hashed_password = hash_password(new_password)
        c.execute('UPDATE users SET password = ? WHERE username = ?', (hashed_password, reset_username))
        conn.commit()
        st.success('Password reset successfully')
        st.session_state['forgot_password'] = False

# Profile management button
if st.session_state['logged_in']:
    if st.button('Manage Profile'):
        st.session_state['manage_profile'] = True

# Profile management section
if 'manage_profile' in st.session_state and st.session_state['manage_profile']:
    st.subheader("Profile")
    st.write(f"Username: {st.session_state['username']}")
    st.write(f"Email: {st.session_state['email']}")
    new_email = st.text_input('Update Email', value=st.session_state['email'])
    if st.button('Update Email'):
        c.execute('UPDATE users SET email = ? WHERE username = ?', (new_email, st.session_state['username']))
        conn.commit()
        st.session_state['email'] = new_email
        st.success('Email updated successfully')


#Close the database connection
conn.close()
if st.session_state['logged_in']: 
# Access AWS credentials from st.secrets
    aws_credentials = st.secrets["aws"]
    access_key_id = aws_credentials["access_key_id"]
    secret_access_key = aws_credentials["secret_access_key"]
    region = aws_credentials["region"]
    bucket_name = aws_credentials["bucket_name"]

# Create a session with the credentials
    session = boto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name=region
    )

# Use the session to create an S3 client
    s3_client = session.client('s3')

# Function to process CSV data and compute Cardan Lumbar angles
    from io import StringIO

    def process_data(file_name_imu00, file_name_imu01):
        # Define the column names for the CSVs
        column_names = ['col1', 'col2', 'col3', 'col4', 'col5', 'Xang', 'Yang', 'Zang', 'quat1', 'quat2', 'quat3', 'quat4']

        # Read the CSV files into DataFrames using file-like objects (StringIO)
        df00 = pd.read_csv(StringIO(file_name_imu00), header=None, names=column_names)
        df01 = pd.read_csv(StringIO(file_name_imu01), header=None, names=column_names)

        # Initialize rotation matrices
        R_trunk_GCS = np.zeros((3, 3))
        R_base_GCS = np.zeros((3, 3))

        # Get the number of rows from both DataFrames
        nRows00, nCols00 = df00.shape
        nRows01, nCols01 = df01.shape
        nRows = min(nRows00, nRows01)

        # Initialize list for storing the Cardan lumbar angles (alpha, beta, gamma)
        Cardan_lumbar = []

        for k in range(nRows):
            alpha00 = -df00.iloc[k, 5]
            beta00 = df00.iloc[k, 6]
            gamma00 = df00.iloc[k, 7]

            alpha01 = -df01.iloc[k, 5]
            beta01 = df01.iloc[k, 6]
            gamma01 = df01.iloc[k, 7]

            # Rotation matrix for trunk (IMU00)
            R_trunk_GCS[0, 0] = np.cos(gamma00) * np.cos(beta00)
            R_trunk_GCS[0, 1] = np.cos(gamma00) * np.sin(beta00) * np.sin(alpha00) + np.sin(gamma00) * np.cos(alpha00)
            R_trunk_GCS[0, 2] = np.sin(gamma00) * np.sin(alpha00) - np.cos(gamma00) * np.sin(beta00) * np.cos(alpha00)

            R_trunk_GCS[1, 0] = -np.sin(gamma00) * np.cos(beta00)
            R_trunk_GCS[1, 1] = np.cos(alpha00) * np.cos(gamma00) - np.sin(gamma00) * np.sin(beta00) * np.sin(alpha00)
            R_trunk_GCS[1, 2] = np.sin(gamma00) * np.sin(beta00) * np.cos(alpha00) + np.cos(gamma00) * np.sin(alpha00)

            R_trunk_GCS[2, 0] = np.sin(beta00)
            R_trunk_GCS[2, 1] = -np.cos(beta00) * np.sin(alpha00)
            R_trunk_GCS[2, 2] = np.cos(beta00) * np.cos(alpha00)

            # Rotation matrix for base (IMU01)
            R_base_GCS[0, 0] = np.cos(gamma01) * np.cos(beta01)
            R_base_GCS[0, 1] = np.cos(gamma01) * np.sin(beta01) * np.sin(alpha01) + np.sin(gamma01) * np.cos(alpha01)
            R_base_GCS[0, 2] = np.sin(gamma01) * np.sin(alpha01) - np.cos(gamma01) * np.sin(beta01) * np.cos(alpha01)

            R_base_GCS[1, 0] = -np.sin(gamma01) * np.cos(beta01)
            R_base_GCS[1, 1] = np.cos(alpha01) * np.cos(gamma01) - np.sin(gamma01) * np.sin(beta01) * np.sin(alpha01)
            R_base_GCS[1, 2] = np.sin(gamma01) * np.sin(beta01) * np.cos(alpha01) + np.cos(gamma01) * np.sin(alpha01)

            R_base_GCS[2, 0] = np.sin(beta01)
            R_base_GCS[2, 1] = -np.cos(beta01) * np.sin(alpha01)
            R_base_GCS[2, 2] = np.cos(beta01) * np.cos(alpha01)

            # Compute relative rotation matrix (trunk -> base)
            R_trunk_base = np.dot(R_trunk_GCS, R_base_GCS.T)

            alpha = np.arctan2(-R_trunk_base[2, 1], R_trunk_base[2, 2])
            beta = np.arcsin(R_trunk_base[2, 0])
            gamma = np.arctan(-R_trunk_base[1, 0] / R_trunk_base[0, 0])

            Cardan_lumbar.append([alpha, beta, gamma])

        Cardan_lumbar = np.array(Cardan_lumbar)

        time = np.linspace(0, (nRows - 1) * 0.01, nRows)
        output_mat = np.hstack([time[:, None], Cardan_lumbar])
        header = ['Time', 'Alpha', 'Beta', 'Gamma']

        # Combine data into a DataFrame
        output_df = pd.DataFrame(output_mat, columns=header)

        return output_df, time, Cardan_lumbar

    # Streamlit UI
    st.title("Lumbar ROM Graphs")
    st.subheader("Select each IMU file from the bucket")

    # Let the user select files from the S3 bucket
    files = s3_client.list_objects_v2(Bucket=bucket_name)['Contents']
    file_names = [file['Key'] for file in files]

    # Allow the user to select two files from the S3 bucket
    uploaded_file_imu00 = st.selectbox("Select IMU00 CSV file", file_names)
    uploaded_file_imu01 = st.selectbox("Select IMU01 CSV file", file_names)

    if uploaded_file_imu00 and uploaded_file_imu01:
        # Download the files from the S3 bucket
        imu00_data = s3_client.get_object(Bucket=bucket_name, Key=uploaded_file_imu00)
        imu01_data = s3_client.get_object(Bucket=bucket_name, Key=uploaded_file_imu01)

        # Read the CSV data into pandas DataFrames
        df_imu00 = imu00_data['Body'].read().decode('utf-8')
        df_imu01 = imu01_data['Body'].read().decode('utf-8')

        # Process the data and get the output DataFrame
        df_output, time, cardan_lumbar = process_data(df_imu00, df_imu01)

        # Plot the results
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot Flexion/Extension (Alpha)
        axs[0].plot(time, cardan_lumbar[:, 0] * 180 / np.pi)  # Convert radians to degrees
        axs[0].set_ylabel('Angle (deg)')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_title('Flexion and Extension')

        # Plot Lateral Flexion (Beta)
        axs[1].plot(time, cardan_lumbar[:, 1] * 180 / np.pi)  # Convert radians to degrees
        axs[1].set_ylabel('Angle (deg)')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_title('Lateral Flexion')

    # Plot Rotation (Gamma)
        axs[2].plot(time, cardan_lumbar[:, 2] * 180 / np.pi)  # Convert radians to degrees
        axs[2].set_ylabel('Angle (deg)')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_title('Rotation')

        # Adjust layout to prevent overlapping titles/labels
        plt.tight_layout()

        # Show the plot in Streamlit
        st.pyplot(fig)

        # Provide option to download the output DataFrame as CSV
        csv_output = df_output.to_csv(index=False)
        st.download_button(
            label="Download Processed Data",
            data=csv_output,
            file_name="processed_lumbar_data.csv",
            mime="text/csv"
        )
