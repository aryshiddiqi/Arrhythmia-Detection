<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\RandomPlotController;
use App\Http\Controllers\FileUploadController;
use App\Http\Controllers\DownloadController;
use Psy\VersionUpdater\Downloader;
use App\Http\Controllers\ExcelController;
use App\Http\Controllers\ModelController;
use App\Http\Controllers\PlotFileController;


Route::get('/', function () {
    return view('upload-form');
});


Route::get('/upload-form', [FileUploadController::class, 'showUploadForm'])->name('upload-form');
Route::post('/upload', [FileUploadController::class, 'upload']);
Route::post('/api/extract', [FileUploadController::class, 'extract']);
Route::get('/download-file/{data}', [DownloadController::class, 'downloadFile'])->name('download.file');
Route::get('/download-data', [DownloadController::class, 'downloadFile'])->name('DownloadController');

Route::post('/load-model', [ExcelController::class, 'loadExcel'])->name('load-model');
Route::post('/predict-data', [ModelController::class, 'predictData'])->name('predict-data');

Route::get('/trained-view', function () {
    return view('trained-view');
})->name('trained-view');

Route::get('/download-file/{file_name}', 'FileController@download')->name('download-file');
Route::post('/excel-data', [ExcelController::class, 'uploadExcel'])->name('excel-data');
Route::get('/plot-file', [PlotFileController::class, 'showPlot'])->name('plot-file');
Route::get('/show-plot', function () {
    return view('show-plot');
})->name('show-plot');

    
