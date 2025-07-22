import 'dart:developer';
import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class ObjectDetection {
  static const String _modelPath = 'assets/best_float322.tflite';
  static const String _labelPath = 'assets/labels.txt';
  
  // YOLO model parameters
  static const int inputSize = 640;
  static const int numClasses = 10;
  static const int numBoxes = 8400; // From your model output (1, 14, 8400)
  static const double confidenceThreshold = 0.5;
  static const double iouThreshold = 0.4;

  Interpreter? _interpreter;
  List<String>? _labels;

  ObjectDetection() {
    _loadModel();
    _loadLabels();
    log('YOLO Object Detection initialized.');
  }

  Future<void> _loadModel() async {
    log('Loading interpreter options...');
    final interpreterOptions = InterpreterOptions();

    // Use XNNPACK Delegate
    if (Platform.isAndroid) {
      interpreterOptions.addDelegate(XNNPackDelegate());
    }

    // // Use Metal Delegate
    // if (Platform.isIOS) {
    //   interpreterOptions.addDelegate(GpuDelegate());
    // }

    log('Loading YOLO interpreter...');
    _interpreter =
        await Interpreter.fromAsset(_modelPath, options: interpreterOptions);
    
    // Print model input/output info
    log('Model input shape: ${_interpreter!.getInputTensor(0).shape}');
    log('Model output shape: ${_interpreter!.getOutputTensor(0).shape}');
  }

  Future<void> _loadLabels() async {
    log('Loading labels...');
    final labelsRaw = await rootBundle.loadString(_labelPath);
    _labels = labelsRaw.split('\n').where((label) => label.trim().isNotEmpty).toList();
    log('Loaded ${_labels!.length} labels: $_labels');
  }

  Uint8List analyseImage(String imagePath) {
    log('Analysing image...');
    // Reading image bytes from file
    final imageData = File(imagePath).readAsBytesSync();

    // Decoding image
    final image = img.decodeImage(imageData);
    if (image == null) {
      log('Failed to decode image');
      return Uint8List(0);
    }

    final originalWidth = image.width;
    final originalHeight = image.height;
    
    log('Original image size: ${originalWidth}x${originalHeight}');

    // Resize image to 640x640 for YOLO model
    final imageInput = img.copyResize(
      image,
      width: inputSize,
      height: inputSize,
    );

    // Create normalized input tensor [1, 640, 640, 3] with values 0-1
    final input = _imageToInputTensor(imageInput);

    final detections = _runInference(input);
    
    // Apply NMS (Non-Maximum Suppression)
    final filteredDetections = _applyNMS(detections);
    
    log('Found ${filteredDetections.length} detections after NMS');

    // Draw detections on original resized image for display
    final outputImage = img.copyResize(image, width: inputSize, height: inputSize);
    _drawDetections(outputImage, filteredDetections);

    log('Done.');
    return img.encodeJpg(outputImage);
  }

  List<List<List<List<double>>>> _imageToInputTensor(img.Image image) {
    // Convert image to normalized tensor [1, 640, 640, 3]
    // YOLO expects RGB values normalized to 0-1
    final input = List.generate(
      1,
      (_) => List.generate(
        inputSize,
        (y) => List.generate(
          inputSize,
          (x) {
            final pixel = image.getPixel(x, y);
            return [
              pixel.r / 255.0, // Red channel normalized
              pixel.g / 255.0, // Green channel normalized
              pixel.b / 255.0, // Blue channel normalized
            ];
          },
        ),
      ),
    );
    return input;
  }

  List<Detection> _runInference(List<List<List<List<double>>>> input) {
    log('Running YOLO inference...');

    // Output shape: [1, 14, 8400] 
    // 14 = 4 (bbox coords) + 10 (class probabilities)
    final output = List.generate(
      1,
      (_) => List.generate(
        14,
        (_) => List.filled(numBoxes, 0.0),
      ),
    );

    _interpreter!.runForMultipleInputs([input], {0: output});
    
    return _parseOutput(output[0]);
  }

  List<Detection> _parseOutput(List<List<double>> output) {
    log('Parsing YOLO output...');
    List<Detection> detections = [];

    // Transpose output from [14, 8400] to [8400, 14] for easier processing
    for (int i = 0; i < numBoxes; i++) {
      // Extract bbox coordinates (first 4 values)
      double centerX = output[0][i];
      double centerY = output[1][i];
      double width = output[2][i];
      double height = output[3][i];

      // Find the class with highest probability (next 10 values)
      double maxProb = 0.0;
      int classId = 0;
      
      for (int j = 0; j < numClasses; j++) {
        double classProb = output[4 + j][i];
        if (classProb > maxProb) {
          maxProb = classProb;
          classId = j;
        }
      }

      // Only keep detections above confidence threshold
      if (maxProb > confidenceThreshold) {
        // Convert from center format to corner format
        double x1 = centerX - width / 2;
        double y1 = centerY - height / 2;
        double x2 = centerX + width / 2;
        double y2 = centerY + height / 2;

        // Scale coordinates to image size (640x640)
        x1 *= inputSize;
        y1 *= inputSize;
        x2 *= inputSize;
        y2 *= inputSize;

        detections.add(Detection(
          classId: classId,
          className: _labels != null && classId < _labels!.length 
              ? _labels![classId] 
              : 'Unknown',
          confidence: maxProb,
          bbox: [x1, y1, x2, y2],
        ));
      }
    }

    log('Found ${detections.length} raw detections');
    return detections;
  }

  List<Detection> _applyNMS(List<Detection> detections) {
    if (detections.isEmpty) return [];

    // Sort by confidence (highest first)
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));

    List<Detection> filteredDetections = [];

    for (int i = 0; i < detections.length; i++) {
      bool keep = true;
      
      for (int j = 0; j < filteredDetections.length; j++) {
        if (detections[i].classId == filteredDetections[j].classId) {
          double iou = _calculateIoU(detections[i].bbox, filteredDetections[j].bbox);
          if (iou > iouThreshold) {
            keep = false;
            break;
          }
        }
      }
      
      if (keep) {
        filteredDetections.add(detections[i]);
      }
    }

    return filteredDetections;
  }

  double _calculateIoU(List<double> box1, List<double> box2) {
    // Calculate intersection over union
    double x1 = math.max(box1[0], box2[0]);
    double y1 = math.max(box1[1], box2[1]);
    double x2 = math.min(box1[2], box2[2]);
    double y2 = math.min(box1[3], box2[3]);

    double intersectionArea = math.max(0, x2 - x1) * math.max(0, y2 - y1);
    
    double box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    double box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    
    double unionArea = box1Area + box2Area - intersectionArea;
    
    return intersectionArea / unionArea;
  }

  void _drawDetections(img.Image image, List<Detection> detections) {
    log('Drawing ${detections.length} detections...');
    
    // Color map for different classes
    final colors = [
      img.ColorRgb8(255, 0, 0),    // Red
      img.ColorRgb8(0, 255, 0),    // Green  
      img.ColorRgb8(0, 0, 255),    // Blue
      img.ColorRgb8(255, 255, 0),  // Yellow
      img.ColorRgb8(255, 0, 255),  // Magenta
      img.ColorRgb8(0, 255, 255),  // Cyan
      img.ColorRgb8(255, 128, 0),  // Orange
      img.ColorRgb8(128, 0, 255),  // Purple
      img.ColorRgb8(255, 192, 203), // Pink
      img.ColorRgb8(128, 128, 128), // Gray
    ];

    for (Detection detection in detections) {
      final color = colors[detection.classId % colors.length];
      
      // Draw bounding box
      img.drawRect(
        image,
        x1: detection.bbox[0].toInt(),
        y1: detection.bbox[1].toInt(),
        x2: detection.bbox[2].toInt(),
        y2: detection.bbox[3].toInt(),
        color: color,
        thickness: 3,
      );

      // Draw label with confidence
      final label = '${detection.className} ${(detection.confidence * 100).toStringAsFixed(1)}%';
      img.drawString(
        image,
        label,
        font: img.arial14,
        x: detection.bbox[0].toInt() + 5,
        y: detection.bbox[1].toInt() + 5,
        color: color,
      );
      
      log('Detected: $label at [${detection.bbox[0].toInt()}, ${detection.bbox[1].toInt()}, ${detection.bbox[2].toInt()}, ${detection.bbox[3].toInt()}]');
    }
  }
}

class Detection {
  final int classId;
  final String className;
  final double confidence;
  final List<double> bbox; // [x1, y1, x2, y2]

  Detection({
    required this.classId,
    required this.className,
    required this.confidence,
    required this.bbox,
  });

  @override
  String toString() {
    return 'Detection(class: $className, confidence: ${confidence.toStringAsFixed(3)}, bbox: $bbox)';
  }
}