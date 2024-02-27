<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use GuzzleHttp\Client;

class ExcelController extends Controller
{
    public function showUploadForm()
    {
        return view('upload-model');
    }

    // Fetch fiture excel
    public function loadExcel(Request $request)
    {
        $url = 'http://127.0.0.1:5000/api/fetch-fitur'; // Sesuaikan dengan alamat API Flask Anda
        
        $file_name = $request->input('file_name');

        // dd($file);

        $response = Http::get($url,['file_name' => $file_name]);

        // dd($response->body());

        // if ($response->hasFile('file_name')) {
        //     $file = $response->file('file_name');
        //     dd('masuk');
        // }

        if ($response->successful()) {
            // Store the file content in a variable
            $fileContent = $response->body();
        
            // Use $fileContent variable for further processing
            // echo "File received. File content length: " . strlen($fileContent);
        } else {
            // Handle errors
            echo "Failed to receive the file.";
        }

        // dd($response);

        $file = $fileContent;
        // dd($file);

        $base64FileContent  = base64_encode(file_get_contents($file_name));

        // Check if $fileContent is not false before using it
        if ($base64FileContent !== false) {
            
            $fileData = [
                'file' => $response->body(),
                'filename' => $file_name,
            ];
            // dd($fileData);

            // $response = Http::withHeaders(['Content-Type' => 'application/json'])
            //                 ->post('http://127.0.0.1:5000/train-model', $fileData);

            $publicPath = storage_path('app/public/');

            $response = Http::timeout(1200)->get('http://127.0.0.1:5000/api/train-model',['file_name' => $file_name, 'public_path' => $publicPath]);
            // dd($response->body());

            

            if ($response->successful()) {
                // Handle success
                $responseData = $response->json();  
                // dd($responseData);
                $success = $responseData['success'];
                $accuracyData = $responseData['accuracy'];
                $modelPath = $responseData['model_path'];
                $classifcation_reports = $responseData['classification_reports'];

                // session()->put('success', true);
                // session()->flush();
                return view('trained-view', ['success' => $success, 'accuracyData' => $accuracyData, 'modelPath' => $modelPath, 'classification_reports' => $classifcation_reports]);



            } else {
                // dd($response->body());
                // Handle failure
                $message = json_decode($response->body(), true);
                // session()->put('error', $message['error']);
                // return redirect()->route('upload-form');
                return redirect()->route('upload-form')->with('error', $message['error']);
            }
        } else {
            return redirect()->route('upload-model.form')->with('error', 'Failed to read file content.');
        }
    }
}
