<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload Excel File</title>
</head>
<body>
    <form action="{{ route('load-model') }}" method="post" enctype="multipart/form-data">
        @csrf
        <label for="excelFile">Choose Excel File:</label>
        <input type="file" name="excelFile" id="excelFile" accept=".xlsx, .xls">

        <button type="submit">Upload</button>
    </form>
    @if(session('success'))
        <div class="alert alert-success">
            {{ session('success') }}
        </div>
    @endif
</body>
</html>
