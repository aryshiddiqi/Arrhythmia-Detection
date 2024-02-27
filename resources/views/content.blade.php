<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Download Page</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #ffffff;
            text-align: center;
            margin: 0;
            padding: 0;
        }

        .container {
            background-color: #f8f9fa;
            padding: 20px;
            border: 1px solid #d6d8db;
            max-width: 600px;
            margin: 50px auto;
            border-radius: 8px;
            /* box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); */
        }

        h1 {
            color: #333;
        }

        p {
            color: #666;
            margin-bottom: 20px;
        }

        a, button {
            text-decoration: none;
            color: #fff;
            padding: 10px 20px;
            border-radius: 5px;
            display: inline-block;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        a {
            background-color: #007bff;
        }

        button {
            background-color: #28a745;
            border: none;
            font-size: 16px;
        }

        a:hover, button:hover {
            /* background-color: #3a3a3a36; */
            filter: brightness(120%);
        }
        .file-list {
            max-width: 400px; 
            margin: 20px auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f8f9fa;
            box-shadow: 0 0 8px rgba(0, 0, 0, 0.1);
        }

        h2 {
            color: #007bff;
            font-size: 18px;
            margin-bottom: 10px;
        }

        .list-group {
            list-style: none;
            padding: 0;
        }

        .list-group-item {
            border: 1px solid #ddd;
            border-radius: 3px;
            margin-bottom: 5px;
            background-color: #fff;
            font-size: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0px;
            border-width: 1px;
        }

        .list-group-item a {
            display: block;
            padding: 8px;
            text-decoration: none;
            color: #333;
            
        }

        .file-text {
            flex: 1; /* Takes up remaining space */
            text-align: center;
        }
        .plot-button{
            font-size: 14px;
            padding: 5px 15px;
            margin: 3px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div>
            <h1>Feature Extraction</h1>
            <p style="text-align: justify; margin-bottom:50px">
                Feature extraction is a critical step in preparing data for machine learning models. It involves deriving meaningful information from the provided data. In this context, various statistical measures are utilized to extract features such as mean heart rate, standard deviation of heart rate, coefficient of variation, root mean square of successive differences (RMSSD), and power spectral density (PSD) components from numeric data. These features play a key role in training machine learning models for further analysis.
                Press the "Train Model" button to initiate the training process using the features extracted from your data.
            </p>
        </div>
        <form action="{{ route('load-model') }}" method="post">
            @csrf
            <input type="hidden" name="file_name" value="{{ $file_name }}">
            <input type="hidden" name="ratio" value="{{ $ratio_data }}">
            <button type="submit" class="btn btn-primary toggle-button">Train Model</button>
        </form>
        
        @if (!$ratio_data)
            <div class="alert alert-warning mt-3" role="alert" style="font-size:10px">
                The data is unbalanced. Consider continuing training with SMOTE.
                <form action="{{ route('load-model') }}" method="post" class="mt-2">
                    @csrf
                    <input type="hidden" name="file_name" value="{{ $file_name }}">
                    <input type="hidden" name="ratio" value="{{ $ratio_data }}">
                    <input type="hidden" name="smote" value="true">
                    <button type="submit" class="btn btn-warning">Continue with SMOTE</button>
                </form>
            </div>
        @endif
        

        <div class="file-list">
            <h2>File List</h2>
            <ul class="list-group">
                @foreach ($files as $file)
                    <li class="list-group-item">
                        <span class="file-text">
                            {{ $file}}
                        </span>
                        <form action="{{ route('plot-file') }}" method="get">
                            @csrf
                            <input type="hidden" name="filename" value="{{$file}}">
                            <input type="hidden" name="file_path_fitur" value="{{ $file_name }}">
                            <button type="submit" class="plot-button" style="margin-left: auto">Plot</button>
                        </form>
                    </li>
                @endforeach
            </ul>
        </div>  
    </div>
</body>
</html>
