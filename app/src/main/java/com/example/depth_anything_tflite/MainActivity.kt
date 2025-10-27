package com.example.depth_anything_tflite

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var depthOverlay: ImageView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var depthEstimator: DepthEstimator

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 🔹 Ambil komponen dari layout
        viewFinder = findViewById(R.id.viewFinder)
        depthOverlay = findViewById(R.id.depthOverlay)

        // 🔹 Inisialisasi DepthEstimator
        depthEstimator = DepthEstimator(this)

        // 🔹 Executor untuk analisis kamera
        cameraExecutor = Executors.newSingleThreadExecutor()

        // 🔹 Minta izin kamera
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
        }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().apply {
                setSurfaceProvider(viewFinder.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .apply {
                    setAnalyzer(cameraExecutor, DepthAnalyzer(depthEstimator, depthOverlay))
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalyzer
                )
            } catch (exc: Exception) {
                Log.e("DepthApp", "❌ Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        // 🔹 Tutup interpreter saat aplikasi berhenti
        if (::depthEstimator.isInitialized) {
            depthEstimator.close()
        }
    }
}

// ==========================================================
// 🔹 DepthAnalyzer: ambil frame kamera → hasilkan depth map
// ==========================================================
private class DepthAnalyzer(
    private val depthEstimator: DepthEstimator,
    private val overlay: ImageView
) : ImageAnalysis.Analyzer {

    override fun analyze(image: ImageProxy) {
        val bitmap = imageProxyToBitmap(image)
        if (bitmap != null) {
            try {
                val depthBitmap = depthEstimator.estimateDepth(bitmap)
                if (depthBitmap != null) {
                    overlay.post {
                        overlay.setImageBitmap(depthBitmap)
                    }
                }
            } catch (e: Exception) {
                Log.e("DepthAnalyzer", "❌ Error saat analisis depth: ${e.message}")
            }
        }
        image.close()
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        return try {
            val yBuffer = image.planes[0].buffer
            val uBuffer = image.planes[1].buffer
            val vBuffer = image.planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = android.graphics.YuvImage(
                nv21,
                android.graphics.ImageFormat.NV21,
                image.width, image.height,
                null
            )

            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(
                android.graphics.Rect(0, 0, image.width, image.height),
                80,
                out
            )
            val imageBytes = out.toByteArray()
            BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        } catch (e: Exception) {
            Log.e("DepthAnalyzer", "❌ Error konversi bitmap: ${e.message}")
            null
        }
    }
}
