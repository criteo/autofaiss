# distributed autofaiss

If you want to generate an index from billion of embeddings, this guide is for you.

This guide is about using pyspark to run autofaiss in multiple nodes.

You may also be interested by [distributed img2dataset](https://github.com/rom1504/img2dataset/blob/main/examples/distributed_img2dataset_tutorial.md)
and [distributed clip inference](https://github.com/rom1504/clip-retrieval/blob/main/docs/distributed_clip_inference.md)

We will be assuming ubuntu 20.04.

## Setup the master node

On the master node:

First download spark:
```bash
wget https://archive.apache.org/dist/spark/spark-3.2.1/spark-3.2.1-bin-hadoop3.2.tgz
tar xf spark-3.2.1-bin-hadoop3.2.tgz
```

Then download autofaiss:
```bash
rm -rf autofaiss.pex
wget https://github.com/criteo/autofaiss/releases/latest/download/autofaiss-3.8.pex -O autofaiss.pex
chmod +x autofaiss.pex
```

If the master node cannot open ports that are visible from your local machine, you can do a tunnel between your local machine and the master node to be able to see the spark ui (at http://localhost:8080)
```bash
ssh -L 8080:localhost:8080 -L 4040:localhost:4040 master_node
```
Replace `master_node` by an ip/host


## Setup the worker nodes

### ssh basic setup

Still in the master node, create a ips.txt with the ips of all the nodes

```bash
ssh-keyscan `cat ips.txt` >> ~/.ssh/known_hosts
```

You may use a script like this to fill your .ssh/config file
```
def generate(ip):
    print(
        f"Host {ip}\n"
        f"        HostName {ip}\n"
        "        User ubuntu\n"
        "        IdentityFile ~/yourkey.pem"
        )

with open("ips.txt") as f:
    lines = f.readlines()
    for line in lines:
        generate(line.strip())
```
python3 generate.py >> ~/.ssh/config

Install pssh with `sudo apt install pssh`

Pick the right username (USER) for the worker nodes, then run this to check your parallel ssh setup:
```bash
USER=ubuntu
```

Optionally, if another node different from the current one has access to the worker nodes, you may need to add a ssh key to all the nodes with:
```
for IP in `cat ips.txt`
do
        ssh-copy-id -i the_new_id_rsa $USER@$IP
done
```

Check you can connect to all the nodes with:
```
parallel-ssh -l $USER -i -h  ips.txt uname -a
```

### Install some packages

```bash
parallel-ssh -l $USER -i -h  ips.txt "sudo apt update"
parallel-ssh -l $USER -i -h  ips.txt "sudo apt install openjdk-11-jre-headless libgl1 htop tmux bwm-ng sshfs python3-distutils python3-apt python3.8 -y"
```


### [Optional] Network setting on aws

On aws, the master node and the worker nodes should be in same VPC and security group and allow inbound, so they can communicate.

### Download autofaiss on all nodes

Download autofaiss on all node by retrying this N times until parallel ssh says success for all:
```bash

parallel-ssh -i -h ips.txt "rm -rf autofaiss.pex"
parallel-ssh -i -h ips.txt "wget https://github.com/criteo/autofaiss/releases/latest/download/autofaiss-3.8.pex -O autofaiss.pex"
parallel-ssh -i -h ips.txt "chmod +x autofaiss.pex"
```

### Download spark on workers

```bash
parallel-ssh -l $USER -i -h  ips.txt  "wget https://archive.apache.org/dist/spark/spark-3.2.1/spark-3.2.1-bin-hadoop3.2.tgz"
parallel-ssh -l $USER -i -h  ips.txt  "tar xf spark-3.2.1-bin-hadoop3.2.tgz"
```

### Start the master node

When you're ready, you can start the master node with:

```bash
./spark-3.2.1-bin-hadoop3.2/sbin/start-master.sh -p 7077
```


### Start the worker nodes

When you're ready, you can start the worker nodes with:

```bash
parallel-ssh -l $USER -i -h  ips.txt  './spark-3.2.1-bin-hadoop3.2/sbin/start-worker.sh -c 16 -m 28G "spark://172.31.35.188:7077"'
```

Replace 172.31.35.188 by the master node ip.


### Stop the worker nodes

When you're done, you can stop the worker nodes with:

```bash
parallel-ssh -l $USER -i -h  ips.txt "rm -rf ~/spark-3.2.1-bin-hadoop3.2/work/*"
parallel-ssh -l $USER -i -h  ips.txt  "pkill java"
```

### Stop the master node

When you're done, you can stop the master node with:

```bash
pkill java
```


### Running autofaiss on it

Once your spark cluster is setup, you're ready to start autofaiss in distributed mode.
Make sure to open your spark UI, at http://localhost:8080 (or the ip where the master node is running)

Save this script to indexing.py.

Then run `./autofaiss.pex indexing.py`

```python
from autofaiss import build_index
from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel

from pyspark import SparkConf, SparkContext

def create_spark_session():
    # this must be a path that is available on all worker nodes
    
    os.environ['PYSPARK_PYTHON'] = "/home/ubuntu/autofaiss.pex"
    spark = (
        SparkSession.builder
        .config("spark.submit.deployMode", "client") \
        .config("spark.executorEnv.PEX_ROOT", "./.pex")
        #.config("spark.executor.cores", "16")
        #.config("spark.cores.max", "48") # you can reduce this number if you want to use only some cores ; if you're using yarn the option name is different, check spark doc
        .config("spark.task.cpus", "16")
        .config("spark.driver.port", "5678")
        .config("spark.driver.blockManager.port", "6678")
        .config("spark.driver.host", "172.31.35.188")
        .config("spark.driver.bindAddress", "172.31.35.188")
        .config("spark.executor.memory", "18G") # make sure to increase this if you're using more cores per executor
        .config("spark.executor.memoryOverhead", "8G")
        .config("spark.task.maxFailures", "100")
        .master("spark://172.31.35.188:7077") # this should point to your master node, if using the tunnelling version, keep this to localhost
        .appName("spark-stats")
        .getOrCreate()
    )
    return spark

spark = create_spark_session()

index, index_infos = build_index(
    embeddings="hdfs://root/path/to/your/embeddings/folder",
    distributed="pyspark",
    file_format="parquet",
    max_index_memory_usage="16G",
    current_memory_available="24G",
    temporary_indices_folder="hdfs://root/tmp/distributed_autofaiss_indices",
    index_path="hdfs://root/path/to/your/index/knn.index",
    index_infos_path="hdfs://root/path/to/your/index/infos.json"
)

```

Another example:

```python
index, index_infos = build_index(
    embeddings=["s3://laion-us-east-1/embeddings/vit-l-14/laion2B-en/img_emb","s3://laion-us-east-1/embeddings/vit-l-14/laion2B-multi/img_emb","s3://laion-us-east-1/embeddings/vit-l-14/laion1B-nolang/img_emb"],
    distributed="pyspark",
    max_index_memory_usage="200G",
    current_memory_available="24G",
    nb_indices_to_keep=10,
    file_format="npy",
    temporary_indices_folder="s3://laion-us-east-1/mytest/my_tmp_folder5",
    index_path="s3://laion-us-east-1/indices/vit-l-14/image/knn.index",
    index_infos_path="s3://laion-us-east-1/indices/vit-l-14/image/infos.json"
)
```

## Benchmark

Computing a 168GB multi pieces `OPQ24_168,IVF131072_HNSW32,PQ24x8` index on 5550336490 embeddings of dim 768 using 10 nodes with 16 cores (c6i.4xlarge) 
takes 6h

