import 'dart:developer';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class ObjectDetection {
  static const String _modelPath = 'assets/best_float32.tflite';
  //custom_ssd_mobilenet_v2_fpn_lite_320x320.tflite
  static const String _labelPath = 'assets/labels.txt';

  Interpreter? _interpreter;
  List<String>? _labels;

  ObjectDetection() {
    _loadModel();
    _loadLabels();
    log('Done.');
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

    log('Loading interpreter...');
    _interpreter =
        await Interpreter.fromAsset(_modelPath, options: interpreterOptions);
  }

  Future<void> _loadLabels() async {
    log('Loading labels...');
    final labelsRaw = await rootBundle.loadString(_labelPath);
    _labels = labelsRaw.split('\n');
  }

  Uint8List analyseImage(String imagePath) {
    log('Analysing image...');
    final imageData = File(imagePath).readAsBytesSync();
    final image = img.decodeImage(imageData);

    final inputSize = 640;
    final resizedImage =
        img.copyResize(image!, width: inputSize, height: inputSize);

    // Normalize to float32 between 0-1
    final imageMatrix = List.generate(inputSize, (y) {
      return List.generate(inputSize, (x) {
        final pixel = resizedImage.getPixel(x, y);
        return [
          pixel.r / 255.0,
          pixel.g / 255.0,
          pixel.b / 255.0,
        ];
      });
    });

    final output = _runInference(imageMatrix);

    final rawOutput = output[0]; // shape: [1, 9, 8400]
    final outputList = rawOutput[0] as List<List<double>>;

    final imageWidth = resizedImage.width;
    final imageHeight = resizedImage.height;

    final threshold = 0.005;
    final boxes = <List<int>>[];
    final classNames = <String>[];
    final scores = <double>[];

    for (int i = 0; i < outputList[0].length; i++) {
      final detection = List<double>.generate(9, (j) => outputList[j][i]);

      final x = detection[0];
      final y = detection[1];
      final w = detection[2];
      final h = detection[3];
      final conf = detection[4];
      print(conf);

      if (conf < threshold) continue;

      final classProbs = detection.sublist(5);
      final maxIndex = classProbs
          .indexWhere((v) => v == classProbs.reduce((a, b) => a > b ? a : b));
      final label = _labels != null && maxIndex < _labels!.length
          ? _labels![maxIndex]
          : 'Unknown';
      log(label);

      final x1 = ((x - w / 2) * imageWidth).toInt().clamp(0, imageWidth - 1);
      final y1 = ((y - h / 2) * imageHeight).toInt().clamp(0, imageHeight - 1);
      final x2 = ((x + w / 2) * imageWidth).toInt().clamp(0, imageWidth - 1);
      final y2 = ((y + h / 2) * imageHeight).toInt().clamp(0, imageHeight - 1);

      boxes.add([x1, y1, x2, y2]);
      classNames.add(label);
      scores.add(conf);
    }

    log('Drawing boxes...');
    for (int i = 0; i < boxes.length; i++) {
      final box = boxes[i];
      img.drawRect(
        resizedImage,
        x1: box[0],
        y1: box[1],
        x2: box[2],
        y2: box[3],
        color: img.ColorRgb8(0, 255, 0),
        thickness: 2,
      );

      img.drawString(
        resizedImage,
        '${classNames[i]} ${(scores[i] * 100).toStringAsFixed(1)}%',
        font: img.arial14,
        x: box[0] + 5,
        y: box[1] + 5,
        color: img.ColorRgb8(0, 255, 0),
      );
    }

    return img.encodeJpg(resizedImage);
  }

 List<List<Object>> _runInference(List<List<List<num>>> imageMatrix) {
  log('Running inference...');

  // Flatten and convert input to Float32List
  final inputBuffer = Float32List(1 * 640 * 640 * 3);
  int index = 0;
  for (var y = 0; y < 640; y++) {
    for (var x = 0; x < 640; x++) {
      for (var c = 0; c < 3; c++) {
        inputBuffer[index++] = imageMatrix[y][x][c].toDouble();
      }
    }
  }

  final input = inputBuffer.buffer.asUint8List(); // only if you use `run`, not `runForMultipleInputs`

  // Build output shape [1, 9, 8400]
  final outputTensor = List.generate(1, (_) => List.generate(9, (_) => List.filled(8400, 0.0)));

  // Use run instead of runForMultipleInputs since it's single input
  _interpreter!.run(inputBuffer.reshape([1, 640, 640, 3]), outputTensor);

  return [outputTensor];
}

}
