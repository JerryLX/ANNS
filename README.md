# Distributed HNSW threshold-sharing probe

这是一个无第三方依赖的单机实验，用多个 HNSW shard 模拟分布式向量检索，并测量节点共享全局 top-k 距离阈值后减少的图搜索工作量。

## 构建与运行

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/hnsw_threshold --n 20000 --queries 200 --shards 4 --k 10 \
  --ef-search 80 --latency-ns 100000
```

程序使用聚类高斯数据，建立多个物理独立的 HNSW shard，并通过全量扫描得到每条 query 的 ground truth。标准输出是 CSV，诊断信息写入标准错误。

## Big ANN Benchmarks 数据

程序直接支持 NeurIPS 2021 Big ANN Benchmarks 的公共二进制格式：

- `.fbin`：float32 向量；
- `.u8bin`：uint8 向量；
- `.i8bin`：int8 向量；
- k-NN ground truth：`uint32(num_queries), uint32(k)`，随后是全部 `uint32` ID 和全部 `float32` 距离。

载入 BIGANN 的示例：

```bash
./build/hnsw_threshold \
  --base bigann_base.u8bin \
  --query bigann_query.bbin \
  --groundtruth bigann_gt.bin \
  --data-type u8 \
  --metric l2 \
  --shards 8 --k 10 --ef-search 100
```

如果文件使用标准 `.u8bin` 扩展名，可以省略 `--data-type`。部分公开文件使用 `.bbin` 等非标准名字，此时需要显式指定类型。

SPACEV 使用 `--data-type i8 --metric l2`；Turing 和 DEEP 使用 `--data-type f32 --metric l2`；Text-to-Image 使用 `--data-type f32 --metric ip`。

调试十亿规模文件时可以只读取前缀：

```bash
./build/hnsw_threshold \
  --base base.1B.fbin --query query.public.100K.fbin \
  --max-base 1000000 --max-queries 1000 --metric l2
```

官方完整数据集的 ground truth 不适用于截断后的 base。因此使用 `--max-base` 时程序不允许同时传入 `--groundtruth`，而会针对载入的前缀执行精确扫描。只截断 query 则仍可使用官方 ground truth。

## 实验组

- `independent`：每个 shard 独立执行标准 HNSW 搜索，最终合并 local top-k。
- `shared`：shard 周期性向协调器发送 local top-k；协调器合并出当前 global top-k threshold，再广播给所有 shard。
- `--latency-ns` 表示一次完整逻辑通信 RTT。模拟器将其平均分到上行和下行。
- `--publish-every` 控制每多少次图节点扩展发布一次候选快照。
- `--threshold-scale` 给共享阈值增加安全余量。`1.0` 最激进；更大的值会多搜索一些图节点，通常能换取更高 Recall。由于程序使用平方 L2 距离，这个参数直接乘在平方距离上。

共享阈值被用于 HNSW 第 0 层的提前终止。由于 HNSW 图并不提供严格的距离下界，这是一条近似剪枝规则，可能影响 Recall；因此必须同时观察计算量和 `recall`，不能只看时延。

## 主要输出

- `avg_distance_computations`：每条 query 在所有 shard 上执行的距离计算总数。
- `avg_expansions`：扩展的第 0 层图节点数。
- `avg_simulated_latency_ns`：按距离计算、扩展开销和临界路径估算的时延。
- `recall`：相对全数据精确 top-k 的 Recall@k。
- `avg_messages`：候选快照和 threshold 广播消息数。
- `avg_threshold_stops`：每条 query 中因共享 threshold 提前停止的 shard 数。

建议先用 `--latency-ns 0` 得到收益上限，再扫描 10 us、50 us、100 us、500 us 和 1 ms。判断优化有效时，应要求 shared 与 independent 的 Recall 接近；若 Recall 明显下降，需要提高 `ef-search` 后比较等 Recall 的计算量。

完整扫描可直接运行：

```bash
bash scripts/run_sweep.sh ./build/hnsw_threshold results.csv
```

例如扫描安全余量：

```bash
for scale in 1.0 1.1 1.25 1.5 2.0; do
  ./build/hnsw_threshold --n 20000 --queries 200 --threshold-scale "$scale"
done
```
