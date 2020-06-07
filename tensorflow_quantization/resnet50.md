# 画像認識モデルの最適化と量子化（ResNet50／ResNet50v1.5）

>NOTE:
>こちらはインテルのTensorFlow用量子化ツールの使い方を日本語訳したものです。オリジナル版は[こちら](https://github.com/IntelAI/models/blob/master/docs/image_recognition/quantization/Tutorial.md)を参照ください。(2020年6月2日現在)

目次:
* [ゴール](#goal)
* [前提条件](#prerequisites)
* [FP32モデルをINT8へ量子化](#floating-point-32-bits-model-quantization-to-8-bits-precision)
* [性能評価](#performance-evaluation)

## ゴール
学習後のモデルの量子化と最適化の目的は以下の通りです。
* モデルのサイズを小さくする。
* オンライン推論を高速に実行する（バッチサイズ = 1）。
* モデル性能の維持（より大きなバッチ推論と精度）。

モバイルアプリケーションやメモリや処理能力に制約のあるシステムの場合に強く推奨します。
通常、性能には多少の損失がありますが、[許容範囲](#性能評価)の範囲内でなければなりません。

その他のリソース。[モバイルとIoTのためのトレーニング後の量子化](https://www.tensorflow.org/lite/performance/post_training_quantization)、および
[TensorFlowグラフ変換ツールユーザーガイド](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms)。

このチュートリアルでは、[インテル® AI 量子化ツール for TensorFlow](https://github.com/IntelAI/tools) を使用して、ResNet50 および ResNet50v1.5 モデルを浮動小数点 32 ビット (FP32) 精度から 8 ビット精度 (INT8) に変換するためのステップ・バイ・ステップ・ガイドを提供します。

## 前提条件
* The binary installed [Intel® optimizations for TensorFlow 2.1.0](https://pypi.org/project/intel-tensorflow/).
```
    $ pip install intel-tensorflow==2.1.0
    $ pip install intel-quantization
```

* The source release repository of [Model Zoo](https://github.com/IntelAI/models) for Intel® Architecture.
```
    $ cd ~
    $ git clone https://github.com/IntelAI/models.git
```
* The source release repository of [Intel® AI Quantization Tools for TensorFlow](https://github.com/IntelAI/tools).
```
    $ cd ~
    $ git clone https://github.com/IntelAI/tools.git
```

* 完全自動量子化には、凍結したFP32事前学習モデルとImageNetデータセットが必要です。
TensorFlowモデルのリポジトリには、以下のものがあります。
[スクリプトと手順](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) を使用して、ImageNet データセットをダウンロード、処理、および TFRecord 形式への変換を行います。


## Floating point 32-bits Model Quantization to 8-bits Precision

このセクションでは、ImageNet データセットが利用可能であることを前提とし、[ResNet50](#resnet50) および [ResNet50v1.5](#resnet50v1.5) のインストラクションで示されているように、FP32 の事前学習モデルをダウンロードすることができます。

インテル® AI 量子化ツール for TensorFlow](https://github.com/IntelAI/tools) リポジトリには、ResNet50 および ResNet50v1.5 の量子化ステップを完全に自動化する python スクリプトがあります。
量子化スクリプトは、事前に学習したモデルの入力パラメータと、ローカル環境に合わせたデータセットパスを必要とします。
そして、モデル名を指定してpythonスクリプトを実行するだけで、FP32からINT8への完全自動量子化を実現します。
```
    $ cd /home/<user>/tools
    $ python api/examples/quantize_model_zoo.py \
        --model model \
        --in_graph /home/<user>/fp32_pretrained_model.pb \
        --out_graph /home/<user>/output.pb \
        --data_location /home/<user>/dataset \
        --models_zoo_location /home/<user>/models
```
`quantize_model_zoo.py` スクリプトは以下の手順でFP32モデルの最適化と量子化を行います。
1) グラフの構造や演算などに基づいてfp32_frozen_graphを最適化する。
2) グラフを量子化します。FP32グラフを出力ノード名を用いてダイナミックレンジINT8グラフに変換します。
3) キャリブレーション 初期量子化されたグラフの動的再量子化範囲(`RequantizationRangeOp`)を静的(定数)に変換します。
4) `RequantizeOp` と量子化された畳み込みを融合し、最終的に最適化されたINT8グラフを生成する。

のような事前に定義されたグラフ量子化パラメータを調整するには、以下のようにします。
(`INPUT_NODE_LIST`, `OUTPUT_NODE_LIST`, `EXCLUDED_NODE_LIST`, `EXCLUDED_NODE_LIST`, `PER_CHANNEL_FLAG`の有効・無効の切り替え)については、[models.json](https://github.com/IntelAI/tools/blob/master/api/config/models.json)ファイルと[量子化APIドキュメント](https://github.com/IntelAI/tools/tree/master/api#integration-with-model-zoo-for-intel-architecture)を確認してください。

## ResNet50

* Download the FP32 ResNet50 pre-trained model to a location of your choice or as suggested:
```
    $ cd /home/<user>/tools/api/models/resnet50
    $ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50_fp32_pretrained_model.pb
```
* 事前学習モデルの入力パラメータ、データセットのパスをローカル環境に合わせて自動量子化スクリプトを実行します。
そうすると、指定された通りに `/home/<user>/tools/api/models/resnet50/resnet50_int8.pb` に保存された、量子化されたResNet50 INT8の事前学習モデルが得られます。
```
    $ cd /home/<user>/tools
    $ python api/examples/quantize_model_zoo.py \
        --model resnet50 \
        --in_graph /home/<user>/tools/api/models/resnet50/resnet50_fp32_pretrained_model.pb \
        --out_graph /home/<user>/tools/api/models/resnet50/resnet50_int8.pb \
        --data_location /home/<user>/imagenet \
        --models_zoo_location /home/<user>/models
```
* グラフ量子化実行終了時のログ出力例。
```
    Model Config: MODEL_NAME:resnet50
    Model Config: LAUNCH_BENCHMARK_PARAMS:{'LAUNCH_BENCHMARK_SCRIPT': 'benchmarks/launch_benchmark.py', 'LAUNCH_BENCHMARK_CMD': ['--model-name resnet50', '--framework tensorflow', '--precision int8', '--mode inference', '--batch-size 100', '--accuracy-only'], 'IN_GRAPH': '--in-graph {}', 'DATA_LOCATION': '--data-location {}'}
    Model Config: QUANTIZE_GRAPH_CONVERTER_PARAMS:{'INPUT_NODE_LIST': ['input'], 'OUTPUT_NODE_LIST': ['predict'], 'EXCLUDED_OPS_LIST': [], 'EXCLUDED_NODE_LIST': [], 'PER_CHANNEL_FLAG': False}
    Model Config: Supported models - ['resnet50', 'resnet50v1_5', 'resnet101', 'mobilenet_v1', 'ssd_mobilenet', 'ssd_resnet34', 'faster_rcnn', 'rfcn', 'inceptionv3']
    Inference Calibration Command: python /home/<user>/models/benchmarks/launch_benchmark.py --model-name resnet50 --framework tensorflow --precision int8 --mode inference --batch-size 100 --accuracy-only --data-location /home/<user>/imagenet --in-graph {}
    ...
    
    ;v0/resnet_v115/conv51/conv2d/Conv2D_eightbit_requant_range__print__;__requant_min_max:[0][2.67806506]
    ;v0/resnet_v115/conv52/conv2d/Conv2D_eightbit_requant_range__print__;__requant_min_max:[0][23.9200363]
    ;v0/mpool0/MaxPool_eightbit_max_v0/conv0/Relu__print__;__max:[5.72005272];v0/mpool0/MaxPool_eightbit_min_v0/conv0/Relu__print__;__min:[-0]
    ...
    
    Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7386, 0.9168)
    Iteration time: 1.3564 ms
    Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7387, 0.9169)
    Iteration time: 1.3461 ms
    Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7387, 0.9169)
    Ran inference with batch size 100
    Log location outside container: /home/<user>/models/benchmarks/common/tensorflow/logs/benchmark_resnet50_inference_int8_20200401_115400.log
    I0401 12:05:21.515716 139714463500096 graph_converter.py:195] Converted graph file is saved to: /home/<user>/output.pb
```

## ResNet50v1.5

* Download the FP32 ResNet50v1.5 pre-trained model to a location of your choice or as suggested:
```
    $ mkdir /home/<user>/tools/api/models/resnet50v1_5 && cd /home/<user>/tools/api/models/resnet50v1_5
    $ wget https://zenodo.org/record/2535873/files/resnet50_v1.pb
```
* Run the automatic quantization script with the input parameters of pre-trained model, dataset path to match with your local environment.
And then, you will get the quantized ResNet50v1.5 INT8 pre-trained model saved in `/home/<user>/tools/api/models/resnet50v1_5/resnet50v1_5_int8.pb` as specified.
```
    $ cd /home/<user>/tools
    $ python api/examples/quantize_model_zoo.py \
        --model resnet50v1_5 \
        --in_graph /home/<user>/tools/api/models/resnet50v1_5/resnet50_v1.pb \
        --out_graph /home/<user>/tools/api/models/resnet50v1_5/resnet50v1_5_int8.pb \
        --data_location /home/<user>/imagenet \
        --models_zoo_location /home/<user>/models
```

* An example for the log output when the graph quantization run completes:
```
Model Config: MODEL_NAME:resnet50v1_5
Model Config: LAUNCH_BENCHMARK_PARAMS:{'LAUNCH_BENCHMARK_SCRIPT': 'benchmarks/launch_benchmark.py', 'LAUNCH_BENCHMARK_CMD': ['--model-name resnet50v1_5', '--framework tensorflow', '--precision int8', '--mode inference', '--batch-size 100', '--accuracy-only'], 'IN_GRAPH': '--in-graph {}', 'DATA_LOCATION': '--data-location {}'}
Model Config: QUANTIZE_GRAPH_CONVERTER_PARAMS:{'INPUT_NODE_LIST': ['input_tensor'], 'OUTPUT_NODE_LIST': ['ArgMax', 'softmax_tensor'], 'EXCLUDED_OPS_LIST': [], 'EXCLUDED_NODE_LIST': [], 'PER_CHANNEL_FLAG': True}
Model Config: Supported models - ['resnet50', 'resnet50v1_5', 'resnet101', 'mobilenet_v1', 'ssd_mobilenet', 'ssd_resnet34', 'faster_rcnn', 'rfcn', 'inceptionv3']
Inference Calibration Command: python /home/<user>/models/benchmarks/launch_benchmark.py --model-name resnet50v1_5 --framework tensorflow --precision int8 --mode inference --batch-size 100 --accuracy-only --data-location /home/<user>/imagenet --in-graph {}
...

;resnet_model/conv2d_5/Conv2D_eightbit_requant_range__print__;__requant_min_max:[0][16.3215694]
;resnet_model/conv2d_6/Conv2D_eightbit_requant_range__print__;__requant_min_max:[0][13.4745159]
;resnet_model/conv2d_7/Conv2D_eightbit_requant_range__print__;__requant_min_max:[0][14.5196199]
...

Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7628, 0.9299)
Iteration time: 1.8439 ms
Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7627, 0.9298)
Iteration time: 1.8366 ms
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7628, 0.9298)
Ran inference with batch size 100
Log location outside container: /home/<user>/models/benchmarks/common/tensorflow/logs/benchmark_resnet50v1_5_inference_int8_20200402_125005.log
I0402 13:07:13.125293 140357697517376 graph_converter.py:195] Converted graph file is saved to: api/models/resnet50v1_5/resnet50v1_5_int8.pb
```

## 性能評価

量子化モデルの性能を検証します。

* 最終的な量子化されたグラフを使用して推論を実行し、精度を計算します。
* 一般的に、精度のターゲットは最適化されたFP32モデルの精度値です。
* 量子化されたINT8グラフの精度は、0.5～1%を超えて低下しないようにしてください。

### ResNet50の精度評価。
IntelAI/models](https://github.com/IntelAI/models)のリポジトリおよび[ResNet50 README](/benchmarks/image_recognition/tensorflow/resnet50/README.md#int8-inference-instructions)を確認してください。
TensorFlowモデルの推論ベンチマークのために、異なる精度での推論が可能です。

#### FP32
[ResNet50 README](/benchmarks/image_recognition/tensorflow/resnet50/README.md#fp32-inference-instructions)の手順に従ってFP32を実行します。
スクリプトを使って `accuracy` を計算し、`--in-graph` で FP32 グラフを利用する。
   ```
        $ cd /home/<user>/models/benchmarks
        $ python launch_benchmark.py \
            --in-graph /home/<user>/tools/api/models/resnet50/resnet50_fp32_pretrained_model.pb \
            --model-name resnet50 \
            --framework tensorflow \
            --precision fp32 \
            --mode inference \
            --accuracy-only \
            --batch-size=100 \
            --socket-id 0 \
            --data-location /home/<user>/imagenet \
            --docker-image intel/intel-optimized-tensorflow:2.1.0
  ```
精度の実行が完了したときのログ出力の末尾は、次のようになります。
   ```
        ...
        Processed 49600 images. (Top1 accuracy, Top5 accuracy) = (0.7422, 0.9184)
        Iteration time: 0.3590 ms
        Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7423, 0.9184)
        Iteration time: 0.3608 ms
        Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7424, 0.9184)
        Iteration time: 0.3555 ms
        Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7425, 0.9185)
        Iteration time: 0.3561 ms
        Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7425, 0.9185)
        ...
   ```

#### INT8

[ResNet50 README](/benchmarks/image_recognition/tensorflow/resnet50/README.md#int8-inference-instructions)の手順に従ってINT8スクリプトを実行して `accuracy` を計算し、`--in-graph` 内の `resnet50_int8.pb` INT8グラフへのパスを使用します。
   ```
        $ cd /home/<user>/models/benchmarks
        $ python launch_benchmark.py \
            --in-graph /home/<user>/tools/api/models/resnet50/resnet50_int8.pb \
            --model-name resnet50 \
            --framework tensorflow \
            --precision int8 \
            --mode inference \
            --accuracy-only \
            --batch-size=100 \
            --socket-id 0 \
            --data-location /home/<user>/imagenet \
            --docker-image intel/intel-optimized-tensorflow:2.1.0
   ```
精度の実行が完了したときのログ出力の末尾は、次のようになります。
   ```
        ...
        Processed 49600 images. (Top1 accuracy, Top5 accuracy) = (0.7369, 0.9159)
        Iteration time: 0.1961 ms
        Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7370, 0.9160)
        Iteration time: 0.1967 ms
        Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7371, 0.9159)
        Iteration time: 0.1952 ms
        Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7371, 0.9160)
        Iteration time: 0.1968 ms
        Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7371, 0.9160)
        ...
   ```


### ResNet50v1.5 Accuracy Evaluation:
Check [IntelAI/models](https://github.com/IntelAI/models) repository and [ResNet50v1.5 README](/benchmarks/image_recognition/tensorflow/resnet50v1_5/README.md#int8-inference-instructions)
for TensorFlow models inference benchmarks with different precisions.

#### FP32
Follow the steps in [ResNet50v1.5 README](/benchmarks/image_recognition/tensorflow/resnet50v1_5/README.md#fp32-inference-instructions) to run the FP32 
script to calculate `accuracy` and use the FP32 graph in `--in-graph`.
   ```
        $ cd /home/<user>/models/benchmarks
        $ python launch_benchmark.py \
            --in-graph /home/<user>/tools/api/models/resnet50v1_5/resnet50_v1.pb \
            --model-name resnet50v1_5 \
            --framework tensorflow \
            --precision fp32 \
            --mode inference \
            --accuracy-only \
            --batch-size=100 \
            --socket-id 0 \
            --data-location /home/<user>/imagenet \
            --docker-image intel/intel-optimized-tensorflow:2.1.0
  ```
The tail of the log output when the accuracy run completes should look something like this:
   ```
        ...
        Processed 49600 images. (Top1 accuracy, Top5 accuracy) = (0.7647, 0.9306)
        Iteration time: 0.4688 ms
        Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7647, 0.9306)
        Iteration time: 0.4694 ms
        Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7648, 0.9307)
        Iteration time: 0.4664 ms
        Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7647, 0.9307)
        Iteration time: 0.4650 ms
        Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7648, 0.9308)
        ...
   ```

#### INT8

Follow the steps in [ResNet50v1.5 README](/benchmarks/image_recognition/tensorflow/resnet50v1_5/README.md#int8-inference-instructions)
to run the INT8 script to calculate `accuracy` and use the path to the `resnet50v1_5_int8.pb` INT8 graph in `--in-graph`.
   ```
        $ cd /home/<user>/models/benchmarks
        $ python launch_benchmark.py \
            --in-graph /home/<user>/tools/api/models/resnet50v1_5/resnet50v1_5_int8.pb \
            --model-name resnet50v1_5 \
            --framework tensorflow \
            --precision int8 \
            --mode inference \
            --accuracy-only \
            --batch-size=100 \
            --socket-id 0 \
            --data-location /home/<user>/imagenet \
            --docker-image intel/intel-optimized-tensorflow:2.1.0
   ```
The tail of the log output when the accuracy run completes should look something like this:
   ```
        ...
        Processed 49600 images. (Top1 accuracy, Top5 accuracy) = (0.7614, 0.9298)
        Iteration time: 0.2126 ms
        Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7614, 0.9298)
        Iteration time: 0.2125 ms
        Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7614, 0.9298)
        Iteration time: 0.2128 ms
        Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7614, 0.9298)
        Iteration time: 0.2122 ms
        Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7616, 0.9298)
       ...
   ```


##
量子化スクリプトやモデルごとの手順の詳細については、[Intel® AI Quantization Tools for TensorFlow](https://github.com/IntelAI/tools/tree/master/api#quantization-python-programming-api-quick-start) を参照してください。また、[Docker サポート](https://github.com/IntelAI/tools/tree/master/api#docker-support) を参照してください。