package com.example.depth_anything_tflite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

class DepthEstimator(context: Context) {

    private var interpreter: Interpreter? = null

    init {
        try {
            val model = FileUtil.loadMappedFile(context, "Depth-Anything-V2_float.tflite")
            val options = Interpreter.Options().apply { setNumThreads(4) }
            interpreter = Interpreter(model, options)
            Log.d("DepthEstimator", "✅ Interpreter siap (CPU mode)")
        } catch (e: Exception) {
            Log.e("DepthEstimator", "❌ Gagal inisialisasi model: ${e.message}")
        }
    }

    fun estimateDepth(bitmap: Bitmap): Bitmap? {
        if (interpreter == null) return null

        val inputW = 256
        val inputH = 256
        val resized = Bitmap.createScaledBitmap(bitmap, inputW, inputH, true)

        // Konversi ke input float32 RGB (0..1)
        val inputBuffer = ByteBuffer.allocateDirect(1 * inputW * inputH * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        for (y in 0 until inputH) {
            for (x in 0 until inputW) {
                val pixel = resized.getPixel(x, y)
                inputBuffer.putFloat(Color.red(pixel) / 255.0f)
                inputBuffer.putFloat(Color.green(pixel) / 255.0f)
                inputBuffer.putFloat(Color.blue(pixel) / 255.0f)
            }
        }
        inputBuffer.rewind()

        // Output buffer (1, h, w)
        val outputShape = interpreter!!.getOutputTensor(0).shape()
        val outH = outputShape[1]
        val outW = outputShape[2]
        val output = Array(1) { Array(outH) { FloatArray(outW) } }

        try {
            interpreter!!.run(inputBuffer, output)
        } catch (e: Exception) {
            Log.e("DepthEstimator", "❌ Error saat inferensi: ${e.message}")
            return null
        }

        val depthBmp = depthArrayToBitmap(output[0], outW, outH)
        return Bitmap.createScaledBitmap(depthBmp, bitmap.width, bitmap.height, true)
    }

    private fun depthArrayToBitmap(depthArray: Array<FloatArray>, width: Int, height: Int): Bitmap {
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // Hitung min & max depth
        var minVal = Float.MAX_VALUE
        var maxVal = Float.MIN_VALUE
        for (y in 0 until height) {
            for (x in 0 until width) {
                val v = depthArray[y][x]
                minVal = min(minVal, v)
                maxVal = max(maxVal, v)
            }
        }

        // Tambah kontras pakai log scale (biar depth lebih kelihatan)
        val range = if (maxVal - minVal == 0f) 1f else (maxVal - minVal)

        for (y in 0 until height) {
            for (x in 0 until width) {
                var v = depthArray[y][x]
                v = (v - minVal) / range // normalisasi 0..1
                v = ln(1f + 9f * v) / ln(10f) // log-scale stretch
                val c = (v * 255).toInt().coerceIn(0, 255)
                bmp.setPixel(x, y, Color.rgb(c, c, c))
            }
        }

        return bmp
    }

    fun close() {
        try {
            interpreter?.close()
            Log.d("DepthEstimator", "🧹 Interpreter ditutup")
        } catch (e: Exception) {
            Log.w("DepthEstimator", "⚠️ Cleanup error: ${e.message}")
        }
    }
}
