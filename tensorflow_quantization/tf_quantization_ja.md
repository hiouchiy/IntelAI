# Quantization Python Programming API Quick Start

>NOTE:
>こちらはインテルの量子化 Python プログラミング API クイックスタートを日本語訳したものです。オリジナル版は[こちら](https://github.com/IntelAI/tools/tree/master/api)を参照ください。(2020年6月3日現在)

目次:
- [量子化 Python プログラミング API クイックスタート](#quantization-python-programming-api-quick-start)
  - [ゴール](#goal)
  - [前提条件](#prerequisites)
  - [Step-by-step 手順 for ResNet-50 量子化](#step-by-step-procedure-for-resnet-50-quantization)
  - [Intel Model Zoo との統合](#integration-with-model-zoo-for-intel-architecture)
  - [ツール](#tools)
    - [Summarize graph](#summarize-graph)
  - [Docker サポート](#docker-support)
  - [FAQ](#faq)


## Goal

Quantization PythonプログラミングAPIの目的は、以下の通りです。
* エントリーを呼び出す量子化ツールを統一する。
* Tensorflow ソースのビルド依存関係を削除します。
* モデルの量子化処理を透明化する。
* 量子化のステップ数を減らす。
* Pythonスクリプトによる推論にシームレスに適応します。

この機能は現在開発中であり、次のリリースではよりインテリジェントな機能が追加される予定です。


## Prerequisites

* TensorFlow 1.14 または 1.15 用のインテル® 最適化がインストールされているバイナリを推奨します。TensorFlow 2.0 用のインテル® 最適化も評価用にサポートされています。

  ```bash
  $ pip install intel-tensorflow==1.15.0
  $ pip install intel-quantization
  ```

* Model Zoo で特定のモデルの量子化を例として実行する場合は、Model Zoo for Intel® アーキテクチャーのソース・リリース・リポジトリが必要です。

  ```bash
  $ cd ~
  $ git clone https://github.com/IntelAI/models.git models && cd models
  $ git checkout v1.5.0
  ```

* TensorFlow 用インテル® AI 量子化ツールのソース・リリース・リポジトリ。

  ```bash
  $ cd ~
  $ git clone https://github.com/IntelAI/tools.git  quantization && cd quantization
  ```

## Step-by-step Procedure for ResNet-50 Quantization

ここでは、完全自動量子化を行うためには、凍結した事前学習モデルとImageNetデータセットが必要となる。

```bash
$ cd ~/quantization/api/models/resnet50
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/resnet50_fp32_pretrained_model.pb
```
TensorFlowモデルリポジトリでは、ImageNetデータセットのダウンロード、処理、TFRecordフォーマットへの変換を行うための[スクリプトと指示書](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data)を提供しています。


1. デモスクリプトの実行

特定のモデルの量子化を `api/examples/` の下で実行するには、Model zoo 用の bash コマンド、カスタムモデル用の bash コマンド、および python プログラミング API の直接呼び出しの 3 つの方法があります。

インテル® アーキテクチャーのModel Zooでモデルを量子化するには、Model zoo用のbashコマンドを使用すると、入力パラメーターが少なくて済む簡単な方法です。 
```bash
$ cd ~/quantization
$ python api/examples/quantize_model_zoo.py \
--model resnet50 \
--in_graph path/to/resnet50_fp32_pretrained_model.pb \
--out_graph path/to/output.pb \
--data_location path/to/imagenet \
--models_zoo_location ~/models
```

事前に学習したモデルの入力パラメータ、データセットのパスをローカル環境に合わせて確認してください。
そして、Pythonスクリプトを実行すると、FP32からINT8への完全自動量子化変換が行われます。


インテル® アーキテクチャーのModel Zooがサポートしていないカスタムモデルについては、別のbashコマンド `api/examples/quantize_cmd.py` が提供されています。
model zoo の bash コマンドとの主な違いは、ユーザーが推論コマンドを準備し、コールバックのパラメーターとして文字列を渡す必要があることです。そして、コールバック関数は途中で生成された一時的なINT8 .pbを実行して、最小と最大のログ情報を出力します。そのため、コールバック関数のコマンド文字列から--input_graphのパラメータと値を削除してください。

```bash
$ python api/examples/quantize_cmd.py \ 
                       --input_graph   path/to/resnet50_fp32_pretrained_model.pb \
                       --output_graph  path/to/output.pb \
                       --callback      inference_command_with_small_subset_for_ min_max_log
                       --inputs 'input'
                       --outputs 'predict'
                       --per_channel False
                       --excluded_ops ''
                       --excluded_nodes ''
```
  `--callback`:学習データの小さなサブセットを用いて推論を実行し、最小値と最大値のログ出力を得る。
  
  `--inputs`:グラフの入力Op名。
  
  `--outputs`:グラフの出力Op名。
  
  `--per_channel`:チャンネル単位の量子化を有効にするかどうかを指定します。チャ
  ンネル単位の量子化は畳み込みカーネルごとにスケールやオフセットが異なります。
  
  `--excluded_ops`:量子化の対象から除外するOpsのリストです。
  
  `--excluded_nodes`:量子化対象から除外するノードのリストです。


3つ目の方法として、PythonプログラミングAPIによる量子化を直接Pythonスクリプトで実行する方法があります。
api/examples/quantize_python.pyにテンプレートが用意されています。キーとなるコードは以下の通りです。

```python
import os
import intel_quantization.graph_converter as converter

def rn50_callback_cmds():
    # This command is to execute the inference with small subset of the training dataset, and get the min and max log output.

if __name__ == '__main__':
    rn50 = converter.GraphConverter('path/to/resnet50_fp32_pretrained_model.pb', None, ['input'], ['predict'])
    # pass an inference script to `gen_calib_data_cmds` to generate calibration data.
    rn50.gen_calib_data_cmds = rn50_callback_cmds()
    # use "debug" option to save temp graph files, default False.
    rn50.debug = True
    rn50.covert()
```

入力された.pbグラフの入出力ノードリストを検出するためのpythonツール[Summarize graph](#summarize-graph)が提供されています。

2. Performance Evaluation

Finally, verify the quantized model performance:
 * Run inference using the final quantized graph and calculate the accuracy.
 * Typically, the accuracy target is the optimized FP32 model accuracy values.
 * The quantized INT8 graph accuracy should not drop more than ~0.5-1%.


Check [Intelai/models](https://github.com/IntelAI/models) repository and [ResNet50 README](https://github.com/IntelAI/models/tree/master/benchmarks/image_recognition/tensorflow/resnet50) for TensorFlow models inference benchmarks with different precisions.

最後に、量子化モデルの性能を検証します。
 * 最終的な量子化されたグラフを使用して推論を実行し、精度を計算します。
 * 一般的に、精度のターゲットは最適化されたFP32モデルの精度値です。
 * 量子化されたINT8グラフの精度は、~0.5-1%を超えて低下してはいけません。


異なる精度のTensorFlowモデル推論ベンチマークについては、[Intelai/models](https://github.com/IntelAI/models)のリポジトリと[ResNet50 README](https://github.com/IntelAI/models/tree/master/benchmarks/image_recognition/tensorflow/resnet50)を参照してください。



## Integration with Model Zoo for Intel Architecture

An integration component with Model Zoo for Intel®  Architecture is provided, that allows users run following models as reference:

インテル® アーキテクチャーのModel Zooとの統合コンポーネントが提供されており、次のモデルをリファレンスとして実行することができます。

- ResNet-50
- ResNet-50 V1_5
- Faster-RCNN
- Inception_V3
- MobileNet_V1
- ResNet-101
- R-FCN
- SSD-MobileNet_V1
- SSD-ResNet34


The model name, launch inference commands for min/max log generation, and specific model quantization parameters are well defined in JSON configuation file `api/config/models.json`.

Take ResNet-50 as an example.

モデル名、min/maxログ生成のための起動推論コマンド、特定のモデルの量子化パラメータは、JSON設定ファイル `api/config/models.json` で十分に定義されている。

ResNet-50を例に説明する。

```
{
  "MODEL_NAME": "resnet50",
  "LAUNCH_BENCHMARK_PARAMS": {
    "LAUNCH_BENCHMARK_SCRIPT": "benchmarks/launch_benchmark.py",
    "LAUNCH_BENCHMARK_CMD": [
      " --model-name=resnet50",
      " --framework=tensorflow",
      " --precision=int8",
      " --mode=inference",
      " --batch-size=100",
      " --accuracy-only"
    ],
    "IN_GRAPH": " --in-graph={}",
    "DATA_LOCATION": " --data-location={}"
  },
  "QUANTIZE_GRAPH_CONVERTER_PARAMS": {
    "INPUT_NODE_LIST": [
      "input"
    ],
    "OUTPUT_NODE_LIST": [
      "predict"
    ],
    "EXCLUDED_OPS_LIST": [],
    "EXCLUDED_NODE_LIST": [],
    "PER_CHANNEL_FLAG": false
  }
}
```

- MODEL_NAME: The model name.

- LAUNCH_BENCHMARK_PARAMS
  - LAUNCH_BENCHMARK_SCRIPT: The relative path of running script in Model Zoo.
  - LAUNCH_BENCHMARK_CMD: The parameters to launch int8 accuracy script in Model Zoo.
  - IN_GRAPH: The path of input graph file.
  - DATA_LOCATION: The path of dataset.
  - MODEL_SOURCE_DIR: The path of tensorflow-models.(optional)
  - DIRECT_PASS_PARAMS_TO_MODEL: The parameters directly passed to the model.(optional)

- QUANTIZE_GRAPH_CONVERTER_PARAMS
  - INPUT_NODE_LIST: The input nodes name list of the model. You can use [Summarize graph](#summarize-graph) to get the inputs and outputs of the graph.
  - OUTPUT_NODE_LIST: The output nodes name list of the model.
  - EXCLUDED_OPS_LIST: The list of operations to be excluded from quantization.
  - EXCLUDED_NODE_LIST: The list of nodes to be excluded from quantization.
  - PER_CHANNEL_FLAG: If set True, enables weight quantization channel-wise.

- MODEL_NAME（モデル名）。モデル名。

- LAUNCH_BENCHMARK_PARAMS
  - LAUNCH_BENCHMARK_SCRIPT: モデル動物園で実行するスクリプトの相対パスです。
  - LAUNCH_BENCHMARK_CMD: モデルZooのint8精度スクリプトを起動するためのパラメータです。
  - IN_GRAPH: 入力グラフファイルのパス
  - DATA_LOCATION: データセットのパス
  - MODEL_SOURCE_DIR: tensorflow-modelsのパス(オプション)
  - DIRECT_PASS_PARAMS_TO_MODEL: モデルに直接渡されるパラメータ．

- quantize_graph_converter_params
  - INPUT_NODE_LIST. モデルの入力ノード名リスト。グラフの入力と出力を取得するには、[Summarize graph](#summarize-graph)を使用します。
  - OUTPUT_NODE_LIST: モデルの出力ノード名リストです。モデルの出力ノード名リスト
  - EXCLUDED_OPS_LIST: モデルの出力ノード名リスト。量子化の対象から除外する演算のリスト。
  - EXCLUDED_NODE_LIST: 量子化対象から除外するノードのリスト。量子化対象から除外するノードのリスト。
  - PER_CHANNEL_FLAG: Trueに設定すると、チャネル単位での重み付け量子化を有効にします。


## Tools

### Summarize graph

In order to remove the TensorFlow source build dependency, the independent Summarize graph tool `api/tools/summarize_graph.py` is provided to dump the possible inputs nodes and outputs nodes of the graph. It could be taken as the reference list for INPUT_NODE_LIST and OUTPUT_NODE_LIST parameters
of graph_converter class. 

TensorFlowソースのビルド依存性を取り除くために、グラフの入力ノードと出力ノードをダンプするための独立したSummarizeグラフツール `api/tools/summarize_graph.py` が提供されています。これは、INPUT_NODE_LISTとOUTPUT_NODE_LISTパラメータの参照リストとして利用できます。
graph_converterクラスの 

- If use graph in binary,

```bash
$ python summarize_graph.py --in_graph=path/to/graph --input_binary
```

- Or use graph in text,

```bash
$ python summarize_graph.py --in_graph=path/to/graph
```


## Docker support 

* [Docker]( https://docs.docker.com/install/ ) - Latest version.

* Build a docker layer which contains Inteli® Optimizations for TensorFlow and Intel® AI Quantization Tools for Tensorflow with the command below. 

* Inteli® Optimizations for TensorFlow とインテル® AI 量子化ツール for Tensorflow を含む docker レイヤーを次のコマンドで構築します。

  ```bash
  $ cd ~/quantization/api/docker
  $ docker build \
       --build-arg HTTP_PROXY=${HTTP_PROXY} \
       --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
       --build-arg http_proxy=${http_proxy} \
       --build-arg https_proxy=${https_proxy} \
       -t quantization:latest -f Dockerfile .
  ```

* Launch quantization script `launch_quantization.py` by providing args as below, this will get user into container environment (`/workspace`) with quantization tools.

  `--docker-image`: Docker image tag from above step (`quantization:latest`).  
  `--in_graph`: Path to your pre-trained model file, which will be mounted inside the container at `/workspace/pretrained_model`.   
  `--out_graph`: When working in the container, all outputs should be saved to `/workspace/output`, so that results are written back to the local machine.  
  `--debug`:Mount the volume and lauch the docker environment to Bash shell environment for debug purpose.   
  `--model_name` and `--models_zoo` are the specific parameters for Model Zoo for Intel® Architecture. If user only want to launch the quantization environment in docker and execute own defined models with `--debug` parameter, both can be skipped. 

* 以下のように引数を指定して量子化スクリプト `launch_quantization.py` を起動すると、量子化ツールのあるコンテナ環境(`/workspace`)にユーザが移動します。

  `--docker-image`. `--docker-image`: 上記のステップで取得したDockerイメージタグ(`quantization:latest`)を指定します。 
  `--in_graph`. このファイルはコンテナ内の `/workspace/pretrained_model` にマウントされます。  
  コンテナ内で作業する場合、すべての出力は `/workspace/pretrained_model` にマウントされます。コンテナ内で作業を行う場合、すべての出力は `/workspace/output` に保存され、結果がローカルマシンに書き戻されるようにします。 
  `--debug`:デバッグのためにボリュームをマウントし、Docker環境をBashシェル環境にラッチします。  
  `--model_name` および `--models_zoo` は、インテル® アーキテクチャーのモデル・ゾーの特定のパラメーターです。ユーザーが docker で量子化環境を起動し、`--debug` パラメータを使って自分で定義したモデルを実行するだけであれば、両方とも省略できます。

* Take the ResNet50 of Model Zoo as an example. 

* モデル動物園のResNet50を例に説明します。

  ```bash
  $ cd ~/quantization/api
  $ python launch_quantization.py  \
  --docker-image quantization:latest \
  --in_graph=/path/to/in_graph.pb \
  --model_name=resnet50 \
  --models_zoo=/path/to/models_zoo \
  --out_graph=/path/to/output.pb \
  --data_location=/path/to/dataset
  ```

## FAQ

* What's the difference with between Quantization Programming APIs and Tensorflow native quantization?
  
  The Quantization Programming APIs are specified for Intel® Optimizations for TensorFlow based on the MKLDNN enabled build. This APIs call the Tensorflow Python models as extension, and provide some special fusion rules, such as, fold_convolutionwithbias_mul, fold_subdivmul_batch_norms, fuse_quantized_conv_and_requantize, mkl_fuse_pad_and_conv,   rerange_quantized_concat etc. 

* 量子化プログラミング API と Tensorflow ネイティブ量子化の違いは？
  
  量子化プログラミング API は、MKLDNN を有効にしたビルドをベースにしたインテル® Optimizations for TensorFlow 用に指定されています。この API は、Tensorflow Python モデルを拡張機能として呼び出し、fold_convolutionwithbias_mul、fold_subdivmul_batch_norms、fuse_quantized_conv_and_requantize、mkl_fuse_pad_and_conv、rerange_quantized_concat などの特殊な融合ルールを提供します。

* How to build the development environment?
  
  For any code contributers, the .whl is easy to be rebuilt to include the specific code for debugging purpose. Refer the build command below.  

* どのように開発環境を構築するのですか?
  
  コードを提供してくれる人は、デバッグ用に特定のコードを含むように .whl をリビルドするのが簡単です。以下のビルドコマンドを参照してください。

  ```bash
  $ cd ~/quantization/api/
  $ python setup.py bdist_wheel
  $ pip install
  $ pip install dist/*.whl
  ```
  


