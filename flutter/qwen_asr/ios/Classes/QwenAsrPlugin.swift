import Flutter
import UIKit
import AVFoundation

public class QwenAsrPlugin: NSObject, FlutterPlugin, FlutterStreamHandler {
    private var audioEngine: AVAudioEngine?
    private var eventSink: FlutterEventSink?
    private var isRecording = false
    
    public static func register(with registrar: FlutterPluginRegistrar) {
        let methodChannel = FlutterMethodChannel(
            name: "qwen_asr/microphone",
            binaryMessenger: registrar.messenger()
        )
        let instance = QwenAsrPlugin()
        registrar.addMethodCallDelegate(instance, channel: methodChannel)
        
        let eventChannel = FlutterEventChannel(
            name: "qwen_asr/microphone_stream",
            binaryMessenger: registrar.messenger()
        )
        eventChannel.setStreamHandler(instance)
    }
    
    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "requestPermission":
            requestPermission(result: result)
        case "start":
            startRecording(call: call, result: result)
        case "stop":
            stopRecording(result: result)
        default:
            result(FlutterMethodNotImplemented)
        }
    }
    
    private func requestPermission(result: @escaping FlutterResult) {
        AVAudioSession.sharedInstance().requestRecordPermission { granted in
            DispatchQueue.main.async {
                result(granted)
            }
        }
    }
    
    private func startRecording(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard !isRecording else {
            result(FlutterError(
                code: "ALREADY_RECORDING",
                message: "Already recording audio",
                details: nil
            ))
            return
        }
        
        // Check permission
        guard AVAudioSession.sharedInstance().recordPermission == .granted else {
            result(FlutterError(
                code: "PERMISSION_DENIED",
                message: "Microphone permission not granted",
                details: nil
            ))
            return
        }
        
        let args = call.arguments as? [String: Any]
        let sampleRate = args?["sampleRate"] as? Int ?? 16000
        
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
            try session.setActive(true)
            
            audioEngine = AVAudioEngine()
            
            let inputNode = audioEngine!.inputNode
            let recordingFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(sampleRate),
                channels: 1,
                interleaved: false
            )!
            
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
                guard let self = self, self.isRecording else { return }
                
                // Convert buffer to array of floats
                let channelData = buffer.floatChannelData![0]
                let frameLength = Int(buffer.frameLength)
                var samples = [Float](repeating: 0, count: frameLength)
                
                for i in 0..<frameLength {
                    samples[i] = channelData[i]
                }
                
                // Send to Flutter on main thread
                DispatchQueue.main.async {
                    self.eventSink?(samples)
                }
            }
            
            audioEngine?.prepare()
            try audioEngine?.start()
            isRecording = true
            
            result(nil)
        } catch {
            result(FlutterError(
                code: "START_FAILED",
                message: "Failed to start recording: \(error.localizedDescription)",
                details: nil
            ))
        }
    }
    
    private func stopRecording(result: @escaping FlutterResult) {
        guard isRecording else {
            result(nil)
            return
        }
        
        isRecording = false
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine = nil
        
        do {
            try AVAudioSession.sharedInstance().setActive(false)
        } catch {
            print("Failed to deactivate audio session: \(error)")
        }
        
        result(nil)
    }
    
    // MARK: - FlutterStreamHandler
    
    public func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        self.eventSink = events
        return nil
    }
    
    public func onCancel(withArguments arguments: Any?) -> FlutterError? {
        self.eventSink = nil
        if isRecording {
            stopRecording { _ in }
        }
        return nil
    }
}
