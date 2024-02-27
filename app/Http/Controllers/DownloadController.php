<?php

namespace App\Http\Controllers;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Facades\View;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\File;
use Illuminate\Support\Facades\Http;

class DownloadController extends Controller
{
    public function downloadFile(Request $request)
    {
        // dd($request->input());
        $file_name = $request->input('data');
        $files = $request->input('files');
        $ratio_data = $request->input('ratio');

        $files = Http::get('http://127.0.0.1:5000/get-files', [
            'folder_path' => $files,
        ]);
        // $files = File::allFiles($files);
        $files = $files->json('files');
        // dd($files);
        // $files = array_slice($files, 0, 10);
        // $file_name = Storage::url($fileName);
        return view('/content', compact('file_name', 'files', 'ratio_data'));
    }
}
