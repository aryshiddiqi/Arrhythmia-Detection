<?php

// app/Http/Controllers/FileController.php

namespace App\Http\Controllers;

use Illuminate\Support\Facades\Storage;

class FileController extends Controller
{
    public function download($file_name)
    {
        $file_path = storage_path("app/public/{$file_name}");

        if (file_exists($file_path)) {
            return response()->download($file_path);
        }

        abort(404, 'File not found');
    }
}
