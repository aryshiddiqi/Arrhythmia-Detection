<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arrhythmia Detection</title>
    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.1.5/jszip.min.js"></script>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f7f7f7;
            margin: 0;
            padding: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
        }

        .container {
            max-width: 600px;
            text-align: center;
        }

        h2 {
            color: #333;
        }

        form {
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
            text-align: center;
        }

        input {
            width: 100%;
            padding: 10px;
            margin-bottom: 20px;
            box-sizing: border-box;
        }

        button {
            background-color: #007bff;
            color: #fff;
            padding: 12px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            width: 100%;
        }

        button:hover {
            background-color: #0056b3;
        }

        #progressContainer {
            display: none;
            margin-top: 20px;
        }

        progress {
            width: 100%;
            margin-bottom: 10px;
        }

        .alert {
            color: #721c24;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
    </style>

    <script>
        function showSessionError() {
            var error = "{!! addslashes(session('error')) !!}";

            if (error) {
                Swal.fire({
                    icon: 'error',
                    title: 'Oops...',
                    text: error,
                });
            }
        }

        window.onload = function() {
            showSessionError();
        };
    </script>
</head>
<body>
    <div style="text-align: center" class="container">
        <h2>Arrhythmia Detection</h2>
        <p style="color:#666">
            Welcome to the Arrhythmia Detection page. This platform allows you to upload a ZIP or RAR file containing known signal data for arrhythmia detection. The system will perform feature extraction on the provided signals, enabling further training of a model based on the extracted features. Use the file input below to upload your data and initiate the process. Upon completion, you'll receive feedback on the progress. Get started by selecting a file and clicking the "Start Extraction" button.
        </p>
        <form method="POST" action="/upload" enctype="multipart/form-data" id="zipForm">
            @csrf
            <input type="file" name="zip_file" id="zipFileInput" accept=".zip, .rar">
            <button type="button submit" onclick="countFiles()">Start Extraction</button>
            <div id="progressContainer">
                <progress id="fileProgress" max="100" value="0"></progress>
                <p id="progressText">0%</p>
            </div>
        </form>
    </div>
    <div id="progress-container"></div>

    <script>
        const progressContainer = document.getElementById('progress-container');
        const eventSource = new EventSource("{{ url('/zip/progress') }}");

        eventSource.onmessage = function (event) {
            const progressMessage = document.createElement('p');
            progressMessage.innerText = event.data;
            progressContainer.appendChild(progressMessage);
        };

        eventSource.onerror = function (event) {
            console.error('EventSource failed:', event);
            eventSource.close();
        };
    </script>

    <script>
        document.addEventListener('DOMContentLoaded', function () {
            async function countFiles() {
                const fileInput = document.getElementById('zipFileInput');

                if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
                    alert('Please select a ZIP file.');
                    return;
                }

                const file = fileInput.files[0];

                try {
                    let fileCount;
                    if (file.name.endsWith('.zip')) {
                        // print('masuk');
                        fileCount = await countZipFiles(file);
                        console.log('fileCount')
                        showProgressBar(fileCount);
                    } else if (file.name.endsWith('.rar')) {
                        fileCount = await countRarFiles(file);
                        console.log('fileCount')
                        showProgressBar(fileCount);
                    } else {
                        alert('Unsupported file format. Only ZIP and RAR are supported.');
                        return;
                    }

                    // Display file count
                } catch (error) {
                    console.error('Count files error:', error);
                }
            }

            async function countZipFiles(file) {
                const zip = await JSZip.loadAsync(file);
                const fileCount = Object.keys(zip.files).length;
                return fileCount;
            }

            async function countRarFiles(file) {
                return new Promise((resolve, reject) => {
                    const reader = new FileReader();

                    reader.onload = async function (e) {
                        const arrayBuffer = e.target.result;

                        try {
                            const RarReader = require('unrar-js');
                            const rar = new RarReader(arrayBuffer);
                            const fileCount = rar.getFileList().length;
                            resolve(fileCount);
                        } catch (error) {
                            reject(error);
                        }
                    };

                    reader.onerror = function (error) {
                        reject(error);
                    };

                    reader.readAsArrayBuffer(file);
                });
            }

            function showProgressBar(totalFiles) {
                const progressContainer = document.getElementById('progressContainer');
                const progressBar = document.getElementById('fileProgress');
                const progressText = document.getElementById('progressText');

                progressBar.value = 0;
                progressText.innerText = '0%';

                progressContainer.style.display = 'block';

                const interval = 0.12 * 1000; 
                const totalTime = totalFiles * interval + 5;

                // Simulate progress with setTimeout
                let currentFile = 0;

                function updateProgress() {
                    currentFile++;
                    const percentCompleted = Math.round((currentFile / totalFiles) * 100);
                    progressBar.value = percentCompleted;
                    progressText.innerText = `${percentCompleted}%`;

                    if (currentFile < totalFiles) {
                        setTimeout(updateProgress, interval);
                    } else {
                        // Hide progress container when complete
                        setTimeout(() => {
                            progressContainer.style.display = 'none';
                        }, interval);
                    }
                }

                setTimeout(updateProgress, interval);
            }

            const button = document.querySelector('button');
            button.addEventListener('click', countFiles);
        });
    </script>
</body>
</html>
