# Distributed ANNS Threshold Sharing Benchmark

这是一个单机分布式 ANNS 实验框架，用多个 shard 验证共享 local top-k distance threshold 能否减少检索计算。当前支持 HNSW 和 SPANN-like 两种引擎、合成数据与 Big ANN Benchmarks 二进制数据、事件模拟与真实多进程运行、CPU 绑核及 shard 索引持久化。

## 构建与测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

程序无第三方运行时依赖。C++ 代码使用 `.clang-format` 中的4空格规则。

## 执行模型

### simulated

`--runtime simulated` 在一个进程中用事件队列模拟 shard 搜索和 threshold 通信延迟，适合快速扫描参数：

```bash
./build/hnsw_threshold \
    --engine hnsw --runtime simulated \
    --n 20000 --queries 200 --shards 4 --k 10 \
    --ef-search 80 --latency-ns 100000
```

### multiprocess

`--runtime multiprocess` 为每个 shard 启动一个进程，每个进程内启动固定数量的 query worker：

```bash
./build/hnsw_threshold \
    --engine spann --runtime multiprocess \
    --n 100000 --queries 1000 \
    --shards 2 --threads-per-shard 8 --k 10
```

- 不同 query 并发执行，不存在 query 间全局 barrier。
- 每个 `(query, shard)` 拥有独立、cache-line 对齐的共享内存槽。
- threshold 槽由所属 shard 单写，其他 shard 原子读取，不使用 mutex。
- worker 绑定到当前 cpuset 中不同的逻辑 CPU。
- 程序要求 `shards × threads-per-shard` 不超过允许使用的 CPU 数。

当前 16 逻辑 CPU 机器推荐 `2 × 8`；384 核服务器可使用 `24 × 16`。

搜索期间只共享 local kth threshold，不共享完整候选列表。每条 query 完成后，父进程收集所有 local top-k，通过 `merge_top()` 得到 global top-k。当前 `avg_latency_ns` 不包含父进程最终结果合并时间。

## Threshold 协议

每个 shard 独立发布：

```text
threshold_bits
candidate_count
```

目标 shard 只读取其他 shard 已经产生至少 k 个候选的槽：

```text
external_tau_i = min(local_kth_j), j != i
```

单分片时没有外部 threshold，因此 shared 与 independent 的工作量必须一致。

实验组：

- `--mode independent`：不读取其他 shard threshold。
- `--mode shared`：启用共享 threshold。
- `--mode all`：依次运行两组，便于对照。

## HNSW

HNSW 使用共享 threshold 收紧第0层 frontier early-stop：

```bash
./build/hnsw_threshold \
    --engine hnsw --runtime multiprocess \
    --n 100000 --queries 1000 \
    --shards 2 --threads-per-shard 8 \
    --k 10 --M 16 --ef-construction 100 --ef-search 80 \
    --threshold-scale 1.5
```

HNSW frontier distance 不是未发现节点的严格下界。共享 threshold 可能提前截断通往近邻的桥接节点，因此必须同时检查 Recall。`--threshold-scale` 越大越保守；任何 Recall 明显下降的计算节省都不应视为有效收益。

建议至少满足：

```text
shared_recall >= independent_recall - 0.005
```

## SPANN-like

SPANN-like 模式使用内存 centroid 和 posting，并通过 posting 覆盖半径计算严格 L2 下界：

```text
lower_bound = max(0, distance(query, centroid) - radius)^2
```

只有 `lower_bound > external_tau` 时才跳过 posting：

```bash
./build/hnsw_threshold \
    --engine spann --runtime multiprocess \
    --n 100000 --queries 1000 \
    --shards 2 --threads-per-shard 8 --k 10 \
    --postings-per-shard 32 --nprobe 16 \
    --kmeans-iterations 3 --io-latency-ns 50000
```

当前实现是用于 threshold 穿刺的 SPANN-like 模型，尚未实现生产 SPANN 的层次化平衡聚类、closure replication、量化、真实 `pread/io_uring` SSD 后端和 query-aware `MaxDistRatio`。`--io-latency-ns` 在多进程模式中通过真实等待模拟单次 posting I/O 延迟。

## Big ANN Benchmarks 数据

支持官方公共格式：

- `.fbin`：float32；
- `.u8bin`：uint8；
- `.i8bin`：int8；
- k-NN GT：`uint32(num_queries), uint32(k)`，随后是全部 `uint32` ID 和全部 `float32` distance。

本地 BIGANN 文件：

```text
data/bigann/learn.100M.u8bin       100M × 128
data/bigann/query.public.10K.u8bin  10K × 128
data/bigann/GT_bigann-10M           10K × top-100
```

`GT_bigann-10M` 对应 base 的前 10M。GT 文件名包含 `10M` 时程序自动设置 `--gt-base-size 10000000`，只加载 100M 文件的前 10M 条。也可以显式指定该参数；若同时传入 `--max-base`，两者必须相等。

如果 `--k` 大于 GT 文件头中的 K，程序会警告并将检索、共享 threshold、最终合并和 Recall 的有效 k 统一截断到 GT 最大值。截断后仍要求 `ef-search >= effective_k`。

没有 GT 时，程序会对载入的 base 做精确扫描；这只适合小规模或 `--max-base` 前缀实验。

## 索引持久化

HNSW 和 SPANN-like 都支持每 shard 独立保存：

```text
HNSW: shard_0.hnsw, shard_1.hnsw, ...
SPANN: shard_0.spann, shard_1.spann, ...
```

模式：

- `--index-mode build`：构建；指定 `--index-dir` 时同时保存。
- `--index-mode load`：直接加载，不读取 base、不重新建索引。
- `--index-mode build-if-missing`：文件齐全时加载，否则构建并保存。

首次构建 BIGANN 10M HNSW：

```bash
./build/hnsw_threshold \
    --engine hnsw --runtime multiprocess \
    --base data/bigann/learn.100M.u8bin \
    --query data/bigann/query.public.10K.u8bin \
    --groundtruth data/bigann/GT_bigann-10M \
    --data-type u8 --metric l2 \
    --shards 2 --threads-per-shard 8 --k 10 \
    --M 16 --ef-construction 100 --ef-search 80 \
    --index-dir data/indexes/bigann_10m_hnsw_s2 \
    --index-mode build
```

后续加载：

```bash
./build/hnsw_threshold \
    --engine hnsw --runtime multiprocess \
    --query data/bigann/query.public.10K.u8bin \
    --groundtruth data/bigann/GT_bigann-10M \
    --data-type u8 --metric l2 \
    --shards 2 --threads-per-shard 8 --k 10 \
    --M 16 --ef-construction 100 --ef-search 80 \
    --index-dir data/indexes/bigann_10m_hnsw_s2 \
    --index-mode load
```

SPANN-like 只需将 `--engine hnsw` 及 HNSW 参数替换为：

```text
--engine spann --postings-per-shard 32 --nprobe 16 --kmeans-iterations 3
```

加载时会校验 magic/version、维度、距离类型、posting 数、图边或 posting member 范围，以及索引总向量数与 GT base 规模是否一致。

当前索引以 float 保存向量，体积大于原始 uint8 数据；后续可增加紧凑 uint8 持久化和 mmap 加载。

## 输出字段

标准输出为 CSV，诊断和构建进度写入标准错误。

- `avg_distance_computations`：所有 shard 每 query 的距离计算总数。
- `avg_expansions`：HNSW 图扩展数；SPANN-like 中等于 posting read 数。
- `avg_latency_ns`：simulated 模式为估算临界路径，multiprocess 模式为最慢 shard worker 的实测 wall-clock。
- `recall`：global top-k 相对 GT 的 Recall@k。
- `avg_threshold_stops`：HNSW 因外部 threshold 提前停止的 shard 数。
- `avg_posting_reads`：SPANN-like posting 读取数。
- `avg_posting_prunes`：SPANN-like 安全跳过的 posting 数。
- `avg_bytes_read`：SPANN-like 逻辑读取字节数。
- `avg_messages`：仅 simulated 事件模式的逻辑消息数。

## 扫描脚本

扫描通信延迟和 threshold scale：

```bash
bash scripts/run_sweep.sh ./build/hnsw_threshold results.csv
```

扫描分片数，默认 `1 2 4 8 16 32`：

```bash
bash scripts/run_shard_sweep.sh \
    ./build/hnsw_threshold shard_results.csv \
    --n 20000 --queries 200 --k 10
```

自定义分片数：

```bash
SHARDS="2 4 8 16 24" \
bash scripts/run_shard_sweep.sh \
    ./build/hnsw_threshold shard_results.csv \
    --runtime multiprocess --threads-per-shard 16
```

脚本生成原始 CSV 和 `_summary.csv`，汇总距离计算、图扩展、时延、posting I/O 节省及 Recall 差值。不同 shard 数对应不同物理分片布局，必须使用分别构建的索引目录。

## 已知限制

- multiprocess 的最终 top-k 在所有 shard 进程结束后由父进程合并，聚合时间未计入 `avg_latency_ns`。
- HNSW threshold early-stop 是近似策略，没有 Recall 保证。
- SPANN-like 的严格球形下界在真实 BIGANN posting 上可能过松，导致没有剪枝收益。
- 多进程索引由父进程构建或加载后通过 `fork` 只读共享；尚未做 NUMA first-touch 和跨 socket 数据放置优化。
- 当前持久化向量为 float，尚未支持紧凑 uint8 mmap 和真实 SSD posting 文件。
