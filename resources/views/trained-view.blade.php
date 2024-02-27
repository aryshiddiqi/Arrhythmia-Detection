<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Upload and Predict</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        .container {
            max-width: 1200px;
            margin: auto;
        }
        .response-box {
            background-color: #f8f9fa;
            padding: 20px;
            border: 1px solid #d6d8db;
            border-radius: 5px;
            margin-top: 20px;
        }
        .alert-box {
            margin-top: 20px;
        }
        .toggle-button {
            margin-top: 20px;
        }
        .form-group{
            margin: 20px;
        }

        .image-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
            max-width: 1200px;
            margin: 20px auto;
            font-size: 12px
        }

        .image-column {
            flex: 0 0 calc(33.33% - 20px);
            margin: 10px;
            text-align: center;
        }

        .image-container img {
            width: 100%;
            max-width: 400px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease-in-out;
        }

        .image-container img:hover {
            transform: scale(1.1);
        }

    </style>
</head>
<body>
    <div class="container">
        @if(session('success'))
            <div class="alert alert-success">
                {{ session('success') }}
            </div>
        @endif

        @if(session('error'))
            <div class="alert alert-danger">
                {{ session('error') }}
            </div>
        @endif

        @if(isset($success) && $success)
            <div class="response-box">
                <div class="alert alert-success">
                    File uploaded and model trained successfully!
                </div>
                <h2 style="font-size: 20px; font-weight: bold; margin-bottom: 10px;">Results: Accuracy and Classification Report</h2>
                <button id="toggleAccuracyButton" class="btn btn-primary toggle-button">Show Results</button>
                <div id="accuracyDetails" style="display: none; padding: 20px; border: 1px solid #ccc; margin-top: 20px;">
                    @if(isset($accuracyData) && isset($classification_reports))
                        <p style="font-weight: bold;">KNN Accuracy: {{ number_format($accuracyData['knn_accuracy']*100, 2) }} %</p>
                        <p style="font-weight: bold;">SVC Accuracy: {{ number_format($accuracyData['svc_accuracy']*100, 2) }} %</p>
                        <p style="font-weight: bold;">Random Forest Accuracy: {{ number_format($accuracyData['rf_accuracy']*100, 2) }} %</p>
                
                        <div style="margin-top: 20px;">
                            <h2 style="font-weight: bold; margin-bottom: 10px; font-size: 20px;">KNN Classification Report</h2>
                            <pre>{{ $classification_reports['knn_classification_report'] }}</pre>
                        </div>
                
                        <div style="margin-top: 20px;">
                            <h2 style="font-weight: bold; margin-bottom: 10px; font-size: 20px;">SVC Classification Report</h2>
                            <pre>{{ $classification_reports['svc_classification_report'] }}</pre>
                        </div>
                
                        <div style="margin-top: 20px;">
                            <h2 style="font-weight: bold; margin-bottom: 10px; font-size: 20px;">Random Forest Classification Report</h2>
                            <pre>{{ $classification_reports['rf_classification_report'] }}</pre>
                        </div>
                    @endif
                </div>
                
                <h2 style="font-size: 20px; font-weight: bold; margin-bottom: 10px; margin-top: 20px;">Predict</h2>                
                <form id="uploadForm" action="{{ route('predict-data') }}" method="POST" enctype="multipart/form-data">
                    @csrf
                    <div class="form-group">
                        <label for="xlsxFile">Upload File:</label>
                        <input type="file" class="form-control-file" id="xlsxFile" name="xlsxFile">
                        <small class="form-text text-muted">Accepted file types: .xlsx, .csv, .xls</small>
                    </div>  
                    <input type="hidden" name="modelPath" value="{{ $modelPath }}">
                    <button type="button" id="uploadBtn" class="btn btn-primary">Upload</button>
                </form>                
                <div id="responseContainer" class="response-box">
                </div>
            </div>
        @else
            <div class="alert-box">
                <div class="alert alert-danger">
                    Failed to train the model. Please try again.
                </div>
            </div>
        @endif
    </div>

    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <script src="//ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"></script>
    <link rel="stylesheet" href="{{asset('resources/css/magnific-popup.css')}}">
    <script src="{{asset('resources/js/jquery.magnific-popup.min.js')}}"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/magnific-popup.js/1.1.0/magnific-popup.min.css">

    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/magnific-popup.js/1.1.0/jquery.magnific-popup.min.js"></script>

    <script>
        $(document).ready(function(){
            $('#uploadBtn').click(function(){
                var formData = new FormData($('#uploadForm')[0]);
    
                $.ajax({
                    url: "{{ route('predict-data') }}",
                    type: 'POST',
                    data: formData,
                    processData: false,
                    contentType: false,
                    dataType: 'json',
                    success: function(response){
                        console.log(response); 
                        
                        // Extract predictions_dict and directory path from the response
                        var predictionsDict = response[0];
                        var directoryPath = response[1];

                        var htmlContent = '<h4 style=\'margin-bottom:20px\'>Prediction Results :</h4>';
                        let votingResults = {};

                        for (var key in predictionsDict) {
                            if (predictionsDict.hasOwnProperty(key)) {
                                var cleanKey = key.replace('.joblib', '');
                                var modifiedValues = predictionsDict[key].map(function(value) {
                                    return value === 'A' ? 'Abnormal' : value === 'N' ? 'Normal' : value;
                                });
                                // Count occurrences of 'Abnormal' and 'Normal'
                                var countAbnormal = modifiedValues.filter(value => value === 'Abnormal').length;
                                var countNormal = modifiedValues.filter(value => value === 'Normal').length;

                                // Determine the majority vote
                                var majorityVote = countAbnormal > countNormal ? 'Arrhythmia symptom detected' : 'No arrhythmia symptom detected';

                                // Store the results for this key
                                votingResults[key] = majorityVote;

                                htmlContent += '<p><strong>' + cleanKey + ' model:</strong> ' + modifiedValues.join(', ') + '</p>';
                            }
                        }

                        if (Object.keys(votingResults).length > 0) {
                            var majorityVote = votingResults[Object.keys(votingResults)[0]];
                            var color = majorityVote === 'Normal' ? 'green' : 'red';

                            htmlContent += '<p style="margin-top:35px; font-weight: bold;">Conclusion: <span style="color: ' + color + ';">' + majorityVote + '</span></p>';
                        }

                        htmlContent += '<style>';
                        htmlContent += '.image-container { text-align: center; }'; 
                        htmlContent += '.image-title { font-size: 12px; text-align: center; }'; 
                        htmlContent += '</style>';
                        htmlContent += '<div class="image-container">';
                        htmlContent += '<div>'; 
                        htmlContent += '<h5>Time Domain Plot</h5>';
                        htmlContent += '<a href="' + '{{ asset('storage/') }}/' + directoryPath + '/tdPlot.png" class="image-preview-link" title="Image Title">';
                        htmlContent += '<img src="' + '{{ asset('storage/') }}/' + directoryPath + '/tdPlot.png" alt="Image" class="image-preview">';
                        htmlContent += '</a>';
                        htmlContent += '</div>'; 
                        htmlContent += '<div>'; 
                        htmlContent += '<h5>Frequency Domain Plot</h5>';
                        htmlContent += '<a href="' + '{{ asset('storage/') }}/' + directoryPath + '/fdPlot.png" class="image-preview-link" title="Image Title">';
                        htmlContent += '<img src="' + '{{ asset('storage/') }}/' + directoryPath + '/fdPlot.png" alt="Image" class="image-preview">';
                        htmlContent += '</a>';
                        htmlContent += '</div>'; 
                        htmlContent += '</div>'; 

                        $('#responseContainer').html(htmlContent);

                        $('.image-preview-link').magnificPopup({
                            type: 'image',
                            closeOnContentClick: true,
                            closeBtnInside: false,
                            fixedContentPos: true,
                            mainClass: 'mfp-no-margins mfp-with-zoom',
                            image: {
                                verticalFit: true
                            },
                            zoom: {
                                enabled: true,
                                duration: 300 
                            }
                        });
                    },
                    error: function(xhr, status, error){
                        console.error(xhr, status, error);
                    }
                });

            });
    
            $('#toggleAccuracyButton').click(function() {
                $('#accuracyDetails').toggle();
            });
        });
    </script>
    
</body>
</html>
