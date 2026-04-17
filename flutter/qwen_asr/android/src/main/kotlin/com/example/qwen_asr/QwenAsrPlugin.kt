package com.example.qwen_asr

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Handler
import android.os.Looper
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import io.flutter.plugin.common.PluginRegistry
import java.util.concurrent.atomic.AtomicBoolean

/** QwenAsrPlugin - Main plugin class with microphone support */
class QwenAsrPlugin : FlutterPlugin, MethodCallHandler, ActivityAware,
    PluginRegistry.RequestPermissionsResultListener {

    private lateinit var methodChannel: MethodChannel
    private var eventChannel: EventChannel? = null
    private var eventSink: EventChannel.EventSink? = null

    private var audioRecord: AudioRecord? = null
    private var recordingThread: Thread? = null
    private val isRecording = AtomicBoolean(false)

    private var activityBinding: ActivityPluginBinding? = null
    private var pendingPermissionResult: Result? = null

    companion object {
        private const val PERMISSION_REQUEST_CODE = 12345
        private const val SAMPLE_RATE = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_FLOAT
    }

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        methodChannel = MethodChannel(binding.binaryMessenger, "qwen_asr/microphone")
        methodChannel.setMethodCallHandler(this)

        eventChannel = EventChannel(binding.binaryMessenger, "qwen_asr/microphone_stream")
        eventChannel?.setStreamHandler(object : EventChannel.StreamHandler {
            override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                eventSink = events
            }

            override fun onCancel(arguments: Any?) {
                eventSink = null
                stopRecording()
            }
        })
    }

    override fun onMethodCall(call: MethodCall, result: Result) {
        when (call.method) {
            "requestPermission" -> handleRequestPermission(result)
            "start" -> handleStartRecording(call, result)
            "stop" -> handleStopRecording(result)
            else -> result.notImplemented()
        }
    }

    private fun handleRequestPermission(result: Result) {
        val activity = activityBinding?.activity
        if (activity == null) {
            result.error("NO_ACTIVITY", "Activity not available", null)
            return
        }

        if (hasRecordAudioPermission()) {
            result.success(true)
            return
        }

        // Request permission
        pendingPermissionResult = result
        ActivityCompat.requestPermissions(
            activity,
            arrayOf(Manifest.permission.RECORD_AUDIO),
            PERMISSION_REQUEST_CODE
        )
    }

    private fun hasRecordAudioPermission(): Boolean {
        val activity = activityBinding?.activity ?: return false
        return ContextCompat.checkSelfPermission(
            activity,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun handleStartRecording(call: MethodCall, result: Result) {
        if (!hasRecordAudioPermission()) {
            result.error("PERMISSION_DENIED", "Microphone permission not granted", null)
            return
        }

        if (isRecording.get()) {
            result.error("ALREADY_RECORDING", "Already recording audio", null)
            return
        }

        val sampleRate = call.argument<Int>("sampleRate") ?: SAMPLE_RATE
        val channels = call.argument<Int>("channels") ?: 1
        val bitsPerSample = call.argument<Int>("bitsPerSample") ?: 32

        try {
            startRecordingInternal(sampleRate, channels, bitsPerSample)
            result.success(null)
        } catch (e: Exception) {
            result.error("START_FAILED", "Failed to start recording: ${e.message}", null)
        }
    }

    private fun startRecordingInternal(sampleRate: Int, channels: Int, bitsPerSample: Int) {
        // Calculate buffer size
        val minBufferSize = AudioRecord.getMinBufferSize(
            sampleRate,
            CHANNEL_CONFIG,
            AUDIO_FORMAT
        )

        // Use larger buffer for smoother recording
        val bufferSize = (minBufferSize * 2).coerceAtLeast(sampleRate * 2)

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            bufferSize
        )

        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            throw RuntimeException("AudioRecord initialization failed")
        }

        isRecording.set(true)
        audioRecord?.startRecording()

        // Start recording thread
        recordingThread = Thread {
            val buffer = FloatArray(sampleRate / 10) // 100ms chunks
            val mainHandler = Handler(Looper.getMainLooper())

            while (isRecording.get()) {
                val readResult = audioRecord?.read(
                    buffer,
                    0,
                    buffer.size,
                    AudioRecord.READ_BLOCKING
                ) ?: 0

                if (readResult > 0) {
                    // Copy the valid samples
                    val samples = buffer.copyOf(readResult)

                    // Send to Flutter on main thread
                    mainHandler.post {
                        eventSink?.success(samples.toList())
                    }
                }
            }
        }.apply {
            name = "AudioRecorderThread"
            start()
        }
    }

    private fun handleStopRecording(result: Result) {
        stopRecording()
        result.success(null)
    }

    private fun stopRecording() {
        isRecording.set(false)

        recordingThread?.join(500)
        recordingThread = null

        audioRecord?.apply {
            stop()
            release()
        }
        audioRecord = null
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ): Boolean {
        if (requestCode == PERMISSION_REQUEST_CODE) {
            val granted = grantResults.isNotEmpty() &&
                    grantResults[0] == PackageManager.PERMISSION_GRANTED
            pendingPermissionResult?.success(granted)
            pendingPermissionResult = null
            return true
        }
        return false
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        methodChannel.setMethodCallHandler(null)
        eventChannel?.setStreamHandler(null)
        stopRecording()
    }

    override fun onAttachedToActivity(binding: ActivityPluginBinding) {
        activityBinding = binding
        binding.addRequestPermissionsResultListener(this)
    }

    override fun onDetachedFromActivityForConfigChanges() {
        activityBinding?.removeRequestPermissionsResultListener(this)
        activityBinding = null
    }

    override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) {
        activityBinding = binding
        binding.addRequestPermissionsResultListener(this)
    }

    override fun onDetachedFromActivity() {
        activityBinding?.removeRequestPermissionsResultListener(this)
        activityBinding = null
    }
}
