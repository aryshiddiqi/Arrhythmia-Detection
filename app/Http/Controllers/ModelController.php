<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use GuzzleHttp\Client;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\View;
use Illuminate\Support\Facades\Response;

class ModelController extends Controller{
    public function predictData(Request $request){

        // dd($request->all());
        // dd($request->input('modelPath'));

        $request->validate([
            'xlsxFile' => 'required|file|mimes:xlsx|max:10240', // Wajib ada, harus file ZIP/RAR, maksimal 10240KB (10MB)
        ]);
        
        
        // dd($request);
        if ($request->hasFile('xlsxFile')){
            $file = $request->file('xlsxFile');
            $folderPath = storage_path('app\\public\\');
            // print($folderPath);
            $fileWoExtension = pathinfo(time() . '_' . $file->getClientOriginalName(), PATHINFO_FILENAME);
            $fileName = time() . '_' . $file->getClientOriginalName();
            $fullPath = $folderPath . '\\' .$fileWoExtension;
            
            $resultMessage = '';

            if (!file_exists($fullPath)) {
                if (mkdir($fullPath, 0777, true)) {
                    $resultMessage = "Folder created successfully at: " . $fullPath;
                } else {
                    $resultMessage = "Failed to create folder at: " . $fullPath;
                }
            } else {
                $resultMessage = "Folder already exists at: " . $fullPath;
            }

            $file->storeAs('public\\' . '\\'.$fileWoExtension.'\\' . $fileName);
            // echo($fullPath);
            // dd($folderPath . '\\' .$fileName);

            $response = Http::post('http://127.0.0.1:5000/api/predict-data', [
                'file_path' => $folderPath .  '\\'.$fileWoExtension.'\\'  .$fileName,
                'model_path' => $request->input('modelPath')
            ]);
            
            
            // dd($response->body());  
            
            // if($response){
                
            // // }
            $data = $response->json();
            // dd($data);

        return response()->json($data);
        }
    }
}