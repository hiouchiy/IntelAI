# Quantization Tools（量子化ツール）

>NOTE:
>こちらはインテルのTensorFlow用量子化ツールの使い方を日本語訳したものです。オリジナル版は[こちら](https://github.com/IntelAI/tools/blob/master/tensorflow_quantization/README.md#quantization-tools)を参照ください。(2020年6月2日現在)

これらのツールはFP32で学習されたTensorFlowグラフをINT8のグラフに変換するのに使用されます。
このドキュメントでは、これらのツールのビルド方法と使用方法について説明します。

## 前提条件

* Ubuntu 18.04
* [Docker](https://docs.docker.com/install/) - 最新バージョン


## 量子化ツールのビルド

  `transform_graph` と `summarize_graph` ツールを含むイメージをビルドします。
  最初のビルドには時間がかかるかもしれませんが、その後のビルドはレイヤーがキャッシュされているので、より速くなります。
   ```
        git clone https://github.com/IntelAI/tools.git
        cd tools/tensorflow_quantization

        docker build \
        --build-arg HTTP_PROXY=${HTTP_PROXY} \
        --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
        --build-arg http_proxy=${http_proxy} \
        --build-arg https_proxy=${https_proxy} \
        -t quantization:latest -f Dockerfile .
   ```
 異なる `BASE_IMAGE_ORG` や `BASE_IMAGE_TAG` に基づいて量子化ツールをビルドするには、`docker build` の際に `--build-args` を追加します。

>NOTE:
>量子化ツールのビルドには、bazelのバージョン >= 0.19.2` と `Tensorflow` が必要です。
   ```
        --build-arg BASE_IMAGE_ORG=<new_base_image_org>
        --build-arg BASE_IMAGE_TAG=<new_base_image_tag>
   ```
## 量子化プロセスを開始

  以下のように引数を指定して量子化スクリプト `launch_quantization.py` を起動します。
  これにより、ユーザは量子化ツールを使ってコンテナ環境(`/workspace/tensorflow/`)に入ることができます。

  - `--docker-image`: Docker image tag from above step (`quantization:latest`)
  - `--pre-trained-model-dir`: 事前に学習したモデルのディレクトリへのパス。
     これはコンテナ内で `/workspace/quantization` にマウントされます。コンテナ内で作業を行う際には、結果がローカルマシンの `pre-trained-model-dir` に書き戻されるように、すべての出力を `/workspace/quantization` に保存する必要があります。
  ```
        python launch_quantization.py \
        --docker-image quantization:latest \
        --pre-trained-model-dir /home/<user>/<pre_trained_model_dir>
  ```

### FP32の最適化済みFrozen Graphへのステップ

このセクションでは、訓練されたモデルを.pbまたは.pbtxt形式で起動していると仮定します。
どちらでも構いません。

 1. トポロジグラフ（モデル [graph_def](https://www.tensorflow.org/guide/extend/model_files#graphdef)）とモデルの重みを含むチェックポイントファイル 
 2. モデルグラフと重みの両方を含む[Frozen Graph](https://www.tensorflow.org/guide/extend/model_files#freezing)

最初のシナリオでは、以下のステップ１、２、３を完了してください。第２のシナリオでフリーズしたグラフがある場合は、ステップ２は必要ありませんので、ステップ１と３のみを完了させてください。

 * **ステップ1**. model graph_def` または `model frozen graph` を用いてグラフの入出力ノード名を取得します.
 * **ステップ2(チェックポイントがある場合)**. モデルフローズングラフを作成するために `model graph_def`, `checkpoint files`, `model frozen graph` を利用します.
 * **ステップ3**. `モデルフローズングラフ`と入出力ノード名を用いて, グラフ構造や操作などに基づいて最適化されたモデルグラフ`を生成します.

以下の手順を実行するには、TensorFlowのルートディレクトリ(dockerコンテナ内の `/workspace/tensorflow`)にいる必要があります。

1. グラフの入力ノード名と出力ノード名を求める。
    ```
        $ bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
         --in_graph=/workspace/quantization/<graph_def_file> \
         --print_structure=false >& model_nodes.txt
    ```
    model_nodes.txtファイルを開き、入力ノードと出力ノードの名前を探します。

2. チェックポイントがある場合は、グラフを凍結します。これにより、チェックポイントの値をグラフ内の定数に変換します。

    * input_graphはモデルトポロジのgraph_defであり、チェックポイントファイルが必要です。
    * 出力ノード名はステップ1で取得したものです。
    * INPUT_GRAPHにはバイナリ形式の `.pb` とテキスト形式の `.pbtxt` のどちらかを指定することができます。
    と `--input_binary` フラグを設定しなければなりません (つまり、`.pb` の入力には True、`.pbtxt` の入力には False を設定します)。
    ```
        $ python tensorflow/python/tools/freeze_graph.py \
         --input_graph /workspace/quantization/<graph_def_file> \
         --output_graph /workspace/quantization/freezed_graph.pb \
         --input_binary False \
         --input_checkpoint /workspace/quantization/<checkpoint_file> \
         --output_node_names OUTPUT_NODE_NAMES
    ```

3. モデルのFrozen Graphを最適化する

    * in_graph` にはモデルのフローズングラフのパスを設定します(ステップ2で取得したもの、元のフローズングラフから始めた場合は元のフローズングラフのパス)。
    * inputs` と `--outputs` はグラフの入出力ノード名です(ステップ1で取得したもの)。
    * モデルのトポロジーに基づいて設定される `--transforms` です．TensorFlow
      トランスフォームリファレンス](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#transform-reference)
      また、[Graph Transforms README](/tensorflow_quantization/graph_transforms/README.md)を参照してください。
      を参照してください。
      
     **Note：** `transform_graph` ツールには、[quantize_weights](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#quantize_weights) と [quantize_nodes](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#quantize_nodes) という変換もあります。これらは使用しないでください。その代わりに、インテルはカスタムの `quantize_graph.py` スクリプトを提供しています (ステップ 5)。
     
    ```
        $ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
         --in_graph=/workspace/quantization/freezed_graph.pb\
         --out_graph=/workspace/quantization/optimized_graph.pb \
         --inputs=INPUT_NODE_NAMES \
         --outputs=OUTPUT_NODE_NAMES \
         --transforms='fold_batch_norms'
    ```

4. 最適化されたグラフ `optimized_graph.pb` を用いて推論を実行し、モデルの精度を確認する。
TensorFlowモデルの推論ベンチマークは、[Intelai/models](https://github.com/IntelAI/models)のリポジトリで確認する。

### INT8量子化ステップ

推論を高速化するためには、グラフをより低い精度で量子化する必要があります。
この節では、前節の出力[FP32 Optimized Frozen Graph](#steps-for-FP32-optimized-frozen-graph)を `Int8` の精度に量子化することを目的とします。

5. 最適化されたグラフ（ステップ3から）を、出力ノード名（ステップ1から）を用いて、より低精度に量子化します。
    ```
        $ python tensorflow/tools/quantization/quantize_graph.py \
         --input=/workspace/quantization/optimized_graph.pb \
         --output=/workspace/quantization/quantized_dynamic_range_graph.pb \
         --output_node_names=OUTPUT_NODE_NAMES \
         --mode=eightbit \
         --intel_cpu_eightbitize=True
    ```

6. 量子化されたグラフを動的な再量子化範囲から静的な再量子化範囲に変換します。
   以下の手順で再量子化範囲を凍結します（キャリブレーションとも呼ばれます）。

    * insert_logging() トランスフォームを使用して、logging op を挿入します。このステップで得られたグラフ(logged_quantized_graph.pb)は、次のステップでモデル較正のためのmin.とmax.の範囲を生成するために使用されます。
        ```
        $ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
         --in_graph=/workspace/quantization/quantized_dynamic_range_graph.pb \
         --out_graph=/workspace/quantization/logged_quantized_graph.pb \
         --transforms='insert_logging(op=RequantizationRange, show_name=true, message="__requant_min_max:")'
        ```

    * キャリブレーションデータを生成する。
        * 前のステップで生成した `logged_quantized_graph.pb` グラフを用いて推論を実行します。これは最小値と最大値のログ出力を得るためにグラフを実行しているだけなので、学習データの小さなサブセットを使用しています。
        * `batch_size` はデータサブセットのサイズに応じて調整する必要があります。
        * 推論の実行中、ログには以下のようなminとmaxの出力が表示されるはずです。
          ```
          ;v0/resnet_v10/conv2/conv2d/Conv2D_eightbit_requant_range__print__;__requant_min_max:[-5.75943518][3.43590856]
          ;v0/resnet_v10/conv1/conv2d/Conv2D_eightbit_requant_range__print__;__requant_min_max:[-3.63552189][5.20797968]
          ;v0/resnet_v10/conv3/conv2d/Conv2D_eightbit_requant_range__print__;__requant_min_max:[-1.44367445][1.50843954]
          ...
          ```

        * 以下の説明では、推論実行から出力されるログファイルを `min_max_log.txt` ファイルとして参照します。
          出力ファイルの全例は、[calibration_data](/tensorflow_quantization/tests/calibration_data)テストファイルを参照してください。
          min_max_log.txt` は、[量子化処理の開始](#start-quantization-process) で指定した場所に保存しておくことをお勧めします。
          これはコンテナ内の `/workspace/quantization` にマウントされます。
    
    * 元の量子化グラフ（ステップ5）のRequantizationRangeOpを、min_max_log.txtファイルを使用してmin.とmax.の定数で置き換えます。
        ```
        $ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
        --in_graph=/workspace/quantizationquantized_dynamic_range_graph.pb \
        --out_graph=/workspace/quantization/freezed_range_graph.pb \
        --transforms='freeze_requantization_ranges(min_max_log_file="/workspace/quantization/min_max_log.txt")'
        ```

7. 必要に応じて量子化済みグラフを最適化する

    * (ステップ6で得られた)量子化された `Int8` グラフと適切な `--transforms` オプションを用いてステップ3を繰り返します.
    

最後に、量子化モデルの性能を検証します。
 * 最終的な量子化されたグラフを使用して推論を実行し、モデルの精度を計算します。
 * 一般的に、精度目標は最適化されたFP32モデルの精度値です。
 * 量子化された `Int8` グラフの精度は、~0.5-1%を超えて低下してはいけません。

  TensorFlowモデルの推論ベンチマークについては、[Intelai/models](https://github.com/IntelAI/models)のリポジトリをチェックしてみてください。

### 例

* [ResNet50](https://github.com/IntelAI/models/blob/master/docs/image_recognition/quantization/Tutorial.md)
