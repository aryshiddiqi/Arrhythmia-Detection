from flask import Flask, send_file, jsonify, request
from werkzeug.utils import secure_filename
from flask_wtf.csrf import CSRFProtect
import requests
import tempfile
import zipfile
import patoolib
import os
import pandas as pd
import matplotlib.pyplot as plt 
import pandas as pd
from train_model_script import train_and_save_models
from joblib import load
import numpy as np
import pandas as pd
import pyhrv.frequency_domain as fd
import pyhrv.time_domain as td
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog

app = Flask(__name__)


def extract_compressed_file(file_to_extract, destination_folder):
    folder_name = os.path.splitext(os.path.basename(file_to_extract))[0]  # Ambil nama file tanpa ekstensi

    destination_folder = os.path.join(destination_folder, folder_name)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    try:
        patoolib.extract_archive(file_to_extract, outdir=destination_folder)
        print(f"File '{file_to_extract}' berhasil diekstrak ke '{destination_folder}'")
        # success_message = f"File '{file_to_extract}' successfully extracted to '{destination_folder}'"
        os.remove(file_to_extract)
        return destination_folder
    except patoolib.util.PatoolError as e:
        # error_message = f"Failed to extract file: {e}"
        print(f"Gagal mengekstrak file: {e}")

    return "File berhasil di ekstrak"

def file_check(file_name, folder_path):
    if os.path.exists(os.path.join(folder_path, file_name)):
        path = folder_path + "\\" + file_name
        result = extract_compressed_file(path, folder_path)
        print(result)
        return jsonify({'file_exists': True, 'message': f'File "{path}" ditemukan.'}), 200
    else:
        return jsonify({'file_exists': False, 'message': f'File "{path}" tidak ditemukan.'}), 404

def read_data(file_path):
    df = pd.read_excel(file_path, header=0)
    s1 = df[df.columns[1]]
    return s1

def check_type(file_name):

    file_name = os.path.basename(file_name)

    parts = file_name.split('_')

    result = parts[0]
    # Check if the result is 'normal' or 'abnormal'
    if result.lower() == 'normal':
        return 'N'
    elif result.lower() == 'abnormal':
        return 'A'
    else:
        return 'Unknown'
    
    
def feature_extraction(data, label='unknown', feature_type='train', image_path=""):
    numeric_data = pd.to_numeric(data, errors='coerce')
    numeric_data = numeric_data.dropna()
    
    mean_value = numeric_data.mean()
    hr_value = 60/mean_value
    std_hr = np.std(numeric_data)
    cvr_value = (std_hr/mean_value*100)
    rmssd_value = np.sqrt(np.mean(np.square(np.diff(numeric_data))))
    
    psd = fd.welch_psd(numeric_data, show = False)
   
    peakPSD = psd['fft_peak']
    lfPeak_value = peakPSD[1]
    hfPeak_value = peakPSD[2]
    
    normPSD = psd['fft_norm']
    lfNorm_value = normPSD[0]
    hfNorm_value = normPSD[1]
    
    lfhf_value = lfNorm_value/hfNorm_value
    
    _sdsd = td.sdsd(numeric_data)
    _sdsd = _sdsd['sdsd']
    
    _nn50 = td.nn50(numeric_data)
    _nn50 = _nn50['nn50']
    

    if image_path:
        fd.plt.savefig(image_path+'\\fdPlot.png')
        plt.clf()
        td_image = time_domain_plot(numeric_data)
        td_image.savefig(image_path+'\\tdPlot.png')
        plt.clf()


        # fd.plt.show()
        # td.plt.savefig(image_path+'\\tdPlot.png')

    if feature_type == 'train':
        return [mean_value, hr_value, std_hr, cvr_value, rmssd_value, _nn50, _sdsd, lfPeak_value, hfPeak_value, lfNorm_value, hfNorm_value, lfhf_value, label]
    elif feature_type == 'predict':
        return {
            'mean_value': mean_value,
            'hr_value': hr_value,
            'std_hr': std_hr,
            'cvr_value': cvr_value,
            'rmssd_value': rmssd_value,
            '_nn50': _nn50,
            '_sdsd': _sdsd,
            'lfPeak_value': lfPeak_value,
            'hfPeak_value': hfPeak_value,
            'lfNorm_value': lfNorm_value,
            'hfNorm_value': hfNorm_value,
            'lfhf_value': lfhf_value,
            'label': label
        }
    elif feature_type == 'plot':
        return fd.plt

    fd.plt.close()
    td.plt.close()


def empty_df():
    # Dictionary with column names
    new_df = [
        'mean_value', 'hr_value', 'std_hr', 'cvr_value', 'rmssd_value', '_nn50', '_sdsd', 'lfPeak_value', 'hfPeak_value', 'lfNorm_value', 'hfNorm_value', 'lfhf_value', 'Label']
    # Convert the set of column names to a list
    columns_list = list(new_df)
    # Create an empty DataFrame with only headers
    new_df = pd.DataFrame(columns=columns_list)
    return new_df

import time

def feature(path, training=True):
    new_df = empty_df()

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.xlsx'):
                start_time = time.time()
                file_path = os.path.join(root, file)
                print("Processing:", file_path)
                
                data = read_data(file_path)
                label = check_type(file)
                fe = feature_extraction(data, label=label)

                new_df.loc[len(new_df)] = fe
                end_time = time.time()

                # error_message = f"Failed to extract file: {file}"

    if len(new_df) == 0:
        if path.endswith('.xlsx'):
            print("Processing:", path)
            
            data = read_data(path)
            label = check_type(path)
            # print(label)
            fe = feature_extraction(data, label=label)

            new_df.loc[len(new_df)] = fe

            print(new_df)

            result = new_df.to_json(orient='records')
            # print(result)
            # result = dict(new_df.values)

            return result
        else:
            return 'No Excel files found in the specified path or its subdirectories!'

    name = os.path.join(path, 'fitur.xlsx')
    new_df.to_excel(name, index=False)
    return name

def count_files(folder_path, prefix):
    # Get all filenames in the folder
    filenames = os.listdir(folder_path)
    
    # Count files with the specified prefix
    count = 0
    for filename in filenames:
        # print(filename)
        if os.path.basename(filename).startswith(prefix):
            count += 1
    
    return count

def data_ratio(folder_path):
    abnormal_count = count_files(folder_path, 'abnormal')
    normal_count = count_files(folder_path, 'normal')
    total = abnormal_count+normal_count

    ratio_normal = normal_count / total if normal_count != 0 else float('inf')
    ratio_abnormal = abnormal_count / total if normal_count != 0 else float('inf')

    ratio = max(ratio_normal, ratio_abnormal)
    print(ratio)


    if 0.5 <= ratio <= 0.60:
        print('True')
        return True
    else :
        return False



@app.route('/api/model', methods=['POST', 'GET'])
def model():
    print('model');
    if 'file_name' not in request.json:
        return jsonify({'error': 'Nama file tidak disediakan'}), 400

    file_name = request.json['file_name']
    folder_path = request.json['file_path']  
    new_folder = file_check(file_name, folder_path)

    folder_name = os.path.splitext(os.path.basename(file_name))[0]  # Ambil nama file tanpa ekstensi
    destination_folder = os.path.join(folder_path, folder_name)
   
    # feature extraction
    feature_file = feature(destination_folder)

    feature_dict = {
        'feature_name': feature_file,
    }

    ratio = data_ratio(destination_folder)

    if ratio:
        feature_dict = {
            'feature_name': feature_file,
            'ratio':ratio
        }
        print(feature_dict)
        return jsonify(feature_dict)    
    else:
        feature_dict = {
            'feature_name': feature_file,
            'ratio':ratio
        }
        print(feature_dict)
        return jsonify(feature_dict)    
    
def pop_dictionary(data_dict, json):
    for key in list(data_dict.keys()):
        if key not in json["selected_features"]:
            data_dict.pop(key, None)
    print(data_dict)
    return [i for i in data_dict.values()]

@app.route('/api/predict-data', methods=['POST'])
def execute_feature_extraction():
    print('\nFeature Extraction.....')
    
    if 'file_path' not in request.json:
        return jsonify({'error': 'File path not provided!'}), 400
    
    file_path = request.json['file_path']
    model_path = request.json['model_path']

    print('file_path :', file_path)
    print('public_path :', model_path)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found!'}), 404

    data = read_data(file_path)

    # read features 
    with open(model_path+'\\selected_features.json', 'r') as file:
        data_json = json.load(file)

    # Call the feature extraction logic function
    feature_file = feature_extraction(data, feature_type='predict', image_path = os.path.dirname(file_path))
    feature_file = pop_dictionary(feature_file, data_json)
    feature_file = np.array(feature_file).reshape(1,-1)

    print(data_json)
    print('\n',feature_file)
    
 
    predictions_dict = {}
    for filename in os.listdir(model_path):
        
        # Check if the file is a model file (adjust the condition as needed)
        if filename.endswith('.joblib'):
            # Load the model
            each_model_path = os.path.join(model_path, filename)
            print('model_1 path: ',model_path)
            loaded_model = load(each_model_path)
            
            # Use the loaded model for predictions or other purposes
            predictions = loaded_model.predict(feature_file)
            print(f"Predictions using {filename}: {predictions}")

            predictions_dict[filename] = predictions.tolist()
    # print(jsonify(predictions_dict, os.path.basename(file_path)))

    return jsonify(predictions_dict, os.path.basename(os.path.dirname(file_path))), 200
    
# def feature_extraction()

import json

@app.route('/api/fetch-fitur', methods=['GET'])
def fetch():
    file_path = request.args.get('file_name')

    # print('file_path :', file_path)

    if file_path:
        return send_file(file_path, as_attachment=True)
    else:
        return 'File not found', 404
    
@app.route('/api/train-model', methods=['GET'])
def train_model():
    try:
        print('\nTrain Model......')
        file_path = request.args.get('file_name')
        public_path = request.args.get('public_path')
        ratio = request.args.get('ratio')

        # Membagi string berdasarkan karakter "\"
        parts = file_path.split('\\')
        print(parts)

        # Mengambil bagian yang diinginkan (di sini: indeks ke-6)
        model_folder = parts[6]
        print('file_path :', file_path)
        print('public_path :', public_path)
        model_save_path = public_path +'models\\'+model_folder
        print('\nmodel_path :', model_save_path)


        df = pd.read_excel(file_path)
        accuracy_data, classification_reports = train_and_save_models(df, label='Label', model_save_path=model_save_path,ratio=ratio)

        response_data = {'success': True, 'accuracy': accuracy_data, 'model_path':model_save_path, 'classification_reports': classification_reports}
        return jsonify(response_data), 200
    except Exception as e:
        print("Error:", str(e))
        return {'error': str(e)}, 500

@app.route('/plot', methods=['GET'])
def plot_file():
    filename = request.args.get('file_name')
    folder_path = request.args.get('file_path_fitur')
    print(filename)
    print(folder_path)

    folder_path = os.path.dirname(folder_path)
    file_path = os.path.join(folder_path, filename)
    print('file to plot:', file_path)

    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    df = pd.read_excel(file_path, header=2)

    sample = list(range(1, len(df.index) + 1))
    signal = df[df.columns[1]]


    # Save the plot to a BytesIO buffer
    filename_base = os.path.splitext(filename)[0]
    save_path = os.path.join(folder_path, filename_base)

    print('save path:', save_path)
    print('file_name:', filename_base)

    raw = raw_signal_plot(sample, signal)
    raw_save = save_plot(raw, 'raw', folder_path, filename_base)

    lendf = len(df.index)

    time_domain = time_domain_plot(signal)
    time_domain_save = save_plot(time_domain, 'timedomain', folder_path, filename_base)

    new_DFT = feature_extraction(signal, feature_type='plot')
    new_DFT_save =  save_plot(new_DFT, 'DFT', folder_path, filename_base)

    # Feature Extraction
    feature_extractions = feature(file_path)

    # print(raw_save, time_domain_save, new_DFT)

    response_data = {
        'success': True,
        'images': {'_2dft': new_DFT_save, '_1time_domain': time_domain_save},
        # 'images': {'_2dft': new_DFT_save, 'raw': raw_save, '_1time_domain': time_domain_save},
        'feature': feature_extractions
    }

    return jsonify(response_data), 200


def DFT(sinyal, clr, judul, fs=1000):
    N = len(sinyal)
    re_X = np.zeros(N)
    im_X = np.zeros(N)
    magX = np.zeros(N)

    # DFT
    for k in range(N):
        for n in range(N):
            re_X[k] += sinyal[n] * np.cos(2 * np.pi * k * n / N)
            im_X[k] -= sinyal[n] * np.sin(2 * np.pi * k * n / N)
        magX[k] = np.sqrt(re_X[k]**2 + im_X[k]**2)

    # Plot
    k = np.arange(0, N//2)
    freq = k * fs / N
    plt.figure(figsize=(12, 8))
    plt.plot(freq, magX[:N//2], color=clr, linewidth=2)
    plt.title(judul, fontweight="bold", size=16)
    plt.xlabel('Frequency (Hz)', size=14)
    plt.ylabel('Magnitude', size=14)
    plt.grid(True)
    return plt

def raw_signal_plot(sample, signal):
    # print('raw signal: ')
    # print(sample, signal)
    # RAW SIGNAL PLOTTING
    plt.figure(figsize=(12, 8))
    plt.plot(sample, signal)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude (mV)')
    plt.title("Raw Signal")
    return plt

def time_domain_plot(signal):
    """
    Plots the time domain representation of a signal.

    Parameters:
    - signal: The input signal.
    - fs: Sampling frequency.
    - title: Title for the plot (default is "Time Domain Plot").
    """
    # Generate time vector
    fs = 1/(sum(signal)/1000)
    t = np.arange(0, len(signal) / fs, 1 / fs)
    # print("time domain: ")
    # print(fs, t , signal)

    # Plot the time domain representation
    plt.plot(t, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title("Time Domain Plot")
    plt.grid(True)
    return plt


def save_plot(plt, plot_type, folder_path, filename_base):
    save_path = os.path.join(folder_path, f'{filename_base}_{plot_type}_plot.png')
    print(save_path)
    public_path = parent_and_file_dir(save_path)
    plt.savefig(save_path, format='png')
    plt.clf()
    return public_path

def parent_and_file_dir(file_path):
    # Get the directory of the file
    folder_location = os.path.dirname(file_path)

    # Get the parent directory name
    parent_directory = os.path.basename(folder_location)

    # Get the desired output
    desired_output = os.path.join(parent_directory, os.path.basename(file_path))

    return desired_output

import fnmatch
@app.route('/get-files', methods=['GET'])
def get_filtered_files(num_normal = 5, num_abnormal = 5):

    folder_path = request.args.get('folder_path')

    all_files = os.listdir(folder_path)

    # Filter normal files
    normal_files = [file for file in all_files if fnmatch.fnmatch(file, 'normal*.xlsx')][:num_normal]

    # Filter abnormal files
    abnormal_files = [file for file in all_files if fnmatch.fnmatch(file, 'abnormal*.xlsx')][:num_abnormal]

    return jsonify({'files' : normal_files + abnormal_files}), 200



if __name__ == '__main__':
    app.run(debug=True)  # Jalankan Flask di mode debug, sesuaikan dengan kebutuhan Anda
