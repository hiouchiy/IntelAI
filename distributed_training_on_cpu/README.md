# ディープラーニング分散学習 (Deep Learning Distributed Training) on CPU
## なぜCPUで分散学習するのか
ディープラーニングにおける学習処理は一般的に非常に重いものが多く、一つのCPU（例えば2ソケット搭載の一台のサーバー）のみでは期待に沿う性能が得られないことがあります。従って、GPGPUのようなアクセラレータを使われる方が多いのが実状です。事実、多くのディープラーニングのモデルにとって、学習処理においてはアクセラレータの方がCPUよりも性能が高いケースが多く（2020年4月時点）、「学習 ≒ アクセラレータ」と理解されている方も多いのではないでしょうか。
ただ、ご存知の通りアクセラレータはお安くはないです。かつ、利用用途がディープラーニングやグラフィック処理などと限られるため、購入したは良いものの、場合によっては持て余してしまい、稼働率がなかなか上がらないという事態も発生しえます。
というわけで、CPUです。学習処理目的でアクセラレータを購入しようとされているなら、少しだけ立ち止まっていただき、まずはお手持ちのCPUを複数個束ねて分散学習してみませんか？というお話です。つまり、1個では性能が足りなくとも、複数個束ねればアクセラレータと同等の性能が実現できます。そのための具体的な手順を以下に記載していこうと思います。
インターネット上で「ディープラーニング　分散学習」で検索すると、GPGPUベースの内容ばかりだったので、CPUベースのものをと思い、ここに記載している次第です。
既存CPU資産、遊休CPU資産を使ってモデルを作りましょう！

## 前提
### ハードウェア
- Intel® Xeon® Scalable processors（Skylake以降）を搭載したサーバー複数台
- 8GB 以上のmemory filled in all DIMM channels i.e >=48 GB
- 25Gbps以上の Ethernet. (or InfiniBand)
- 1TB以上のSSDストレージ
### ソフトウェア
- Ubuntu 18.04 (CentOS 7.7でもOK)
- TensorFlow (intel-tensorflow==1.15.2)
- Horovod (horovod == 0.16.4)
- Openmpi (latest from linux APT/YUM repositories)
- Docker (latest from linux APT/YUM repositories)

## 手順
本手順では、Dockerコンテナを用います。なお、複数台のサーバーのうち、1台をマスターノード、それ以外をSlaveノードとして作業を進めてください。
1. Dockerのインストール：全ノード（参照元は[ここ](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04)）
    - sudo apt update
    - sudo apt install apt-transport-https ca-certificates curl software-properties-common
    - curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    - sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
    - sudo apt update
    - apt-cache policy docker-ce
    - sudo apt install docker-ce
    - sudo usermod -aG docker ${USER}
    - su - ${USER}
    - id -nG
    - sudo usermod -aG docker ${USER}
1. Docker Imageをダウンロード：全ノード
    - docker pull vdevaram/hvd_tf_1.15:1.0
1. SSHの設定：Slaveノード
    - vim /etc/ssh/sshd_config
    - PermitRootLogin yes と変更
1. SSHの設定：Masterノード
    - su root
    - ssh-keygen -t rsa
    - Slaveの数だけ下記コマンドを実行
        - ssh-copy-id -i /root/.ssh/id_rsa.pub <user>@<slave_ip>
1. Training Scriptの準備：全ノード
    - とりあえずtf_cnn_benchmarkをダウンロード
        - export WORKSPACE=$HOME/workspace
        - mkdir -p $HOME/workspace
        - git clone https://github.com/tensorflow/benchmarks.git
        - cd benchmarks
        - git checkout cnn_tf_v1.15_compatible
1. Dockerコンテナを実行：全ノード
    - docker run -v $WORKSPACE:/workspace -v /root/.ssh:/root/.ssh --network=host --privileged -it vdevaram/hvd_tf_1.15:1.0
1. SSH Daemonを起動：Slave ノード
    - /usr/sbin/sshd -p 12345; sleep infinity
1. Training Jobを実行：Masterノード
    - HOROVOD_FUSION_THRESHOLD=134217728 mpirun -np 4 --map-by ppr:1:socket:pe=20 --allow-run-as-root --mca plm_rsh_args "-p 12345" -mca btl_tcp_if_include bond0.123 -mca btl ^openib -mca pml ob1 -H <MasterノードのIPアドレス>:9999,<SlaveノードのIPアドレス>:9999 --oversubscribe --report-bindings -x LD_LIBRARY_PATH -x HOROVOD_FUSION_THRESHOLD -x OMP_NUM_THREADS=9 -x KMP_BLOCKTIME=1 -x KMP_AFFINITY=granularity=fine,verbose,compact,1,0 python3 -u /workspace/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --batch_size 64 --num_batches 40 --distortions=False --num_intra_threads 10 --num_inter_threads 1 --local_parameter_device cpu --variable_update horovod --horovod_device cpu

## Kubernetes上での実行方法
製作中．．．


