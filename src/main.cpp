#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

using Vec = std::vector<float>;

struct Args {
    int n = 20000, queries = 200, dim = 32, shards = 4, k = 10;
    int M = 16, ef_construction = 100, ef_search = 80, clusters = 32;
    int publish_every = 1;
    int postings_per_shard = 32, nprobe = 16, kmeans_iterations = 3;
    int threads_per_shard = 1;
    uint64_t seed = 42, latency_ns = 0, distance_ns = 50, expansion_ns = 100;
    uint64_t io_latency_ns = 50000;
    double ssd_bytes_per_ns = 3.0;
    double threshold_scale = 1.0;
    std::string mode = "all", engine = "hnsw", runtime = "simulated";
    std::string index_dir, index_mode = "build";
    std::string base_path, query_path, gt_path, data_type = "auto", metric = "l2";
    uint64_t max_base = 0, max_queries = 0;
    uint64_t gt_base_size = 0;
};

static void usage(const char *p) {
    std::cout << "Usage: " << p << " [options]\n"
              << "  --n N --queries N --dim D --shards N --k K\n"
              << "  --M N --ef-construction N --ef-search N --clusters N\n"
              << "  --mode all|independent|shared --latency-ns N\n"
              << "  --distance-ns N --expansion-ns N --publish-every N\n"
              << "  --threshold-scale F --seed N\n"
              << "  --base FILE --query FILE [--groundtruth FILE]\n"
              << "  --data-type auto|f32|u8|i8 --metric l2|ip\n"
              << "  --max-base N --max-queries N\n"
              << "  --engine hnsw|spann --postings-per-shard N --nprobe N\n"
              << "  --kmeans-iterations N --io-latency-ns N --ssd-bytes-per-ns F\n";
    std::cout << "  --runtime simulated|multiprocess --threads-per-shard N\n"
              << "  --gt-base-size N (or auto-detect 10M from GT filename)\n"
              << "  --index-dir PATH --index-mode build|load|build-if-missing\n";
}

static Args parse_args(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc)
            throw std::runtime_error("missing value for " + key);
        std::string v = argv[++i];
        auto iv = [&] { return std::stoi(v); };
        auto uv = [&] { return std::stoull(v); };
        if (key == "--n")
            a.n = iv();
        else if (key == "--queries")
            a.queries = iv();
        else if (key == "--dim")
            a.dim = iv();
        else if (key == "--shards")
            a.shards = iv();
        else if (key == "--k")
            a.k = iv();
        else if (key == "--M")
            a.M = iv();
        else if (key == "--ef-construction")
            a.ef_construction = iv();
        else if (key == "--ef-search")
            a.ef_search = iv();
        else if (key == "--clusters")
            a.clusters = iv();
        else if (key == "--publish-every")
            a.publish_every = iv();
        else if (key == "--seed")
            a.seed = uv();
        else if (key == "--latency-ns")
            a.latency_ns = uv();
        else if (key == "--distance-ns")
            a.distance_ns = uv();
        else if (key == "--expansion-ns")
            a.expansion_ns = uv();
        else if (key == "--mode")
            a.mode = v;
        else if (key == "--threshold-scale")
            a.threshold_scale = std::stod(v);
        else if (key == "--engine")
            a.engine = v;
        else if (key == "--runtime")
            a.runtime = v;
        else if (key == "--threads-per-shard")
            a.threads_per_shard = iv();
        else if (key == "--index-dir")
            a.index_dir = v;
        else if (key == "--index-mode")
            a.index_mode = v;
        else if (key == "--postings-per-shard")
            a.postings_per_shard = iv();
        else if (key == "--nprobe")
            a.nprobe = iv();
        else if (key == "--kmeans-iterations")
            a.kmeans_iterations = iv();
        else if (key == "--io-latency-ns")
            a.io_latency_ns = uv();
        else if (key == "--ssd-bytes-per-ns")
            a.ssd_bytes_per_ns = std::stod(v);
        else if (key == "--base")
            a.base_path = v;
        else if (key == "--query")
            a.query_path = v;
        else if (key == "--groundtruth")
            a.gt_path = v;
        else if (key == "--data-type")
            a.data_type = v;
        else if (key == "--metric")
            a.metric = v;
        else if (key == "--max-base")
            a.max_base = uv();
        else if (key == "--max-queries")
            a.max_queries = uv();
        else if (key == "--gt-base-size")
            a.gt_base_size = uv();
        else
            throw std::runtime_error("unknown option: " + key);
    }
    if (a.n <= 0 || a.queries <= 0 || a.dim <= 0 || a.shards <= 0 || a.k <= 0 || a.M <= 1 || a.ef_search <= 0 ||
        a.ef_construction < a.M || a.publish_every <= 0 || a.clusters <= 0 || a.threshold_scale < 1.0)
        throw std::runtime_error("invalid arguments (ef-search must be >= k)");
    if (a.mode != "all" && a.mode != "independent" && a.mode != "shared")
        throw std::runtime_error("mode must be all, independent, or shared");
    if (a.metric != "l2" && a.metric != "ip")
        throw std::runtime_error("metric must be l2 or ip");
    if (a.data_type != "auto" && a.data_type != "f32" && a.data_type != "u8" && a.data_type != "i8")
        throw std::runtime_error("data-type must be auto, f32, u8, or i8");
    if (a.base_path.empty() != a.query_path.empty() && !(a.index_mode == "load" && !a.query_path.empty()))
        throw std::runtime_error("--base and --query must be provided together");
    if (a.engine != "hnsw" && a.engine != "spann")
        throw std::runtime_error("engine must be hnsw or spann");
    if (a.runtime != "simulated" && a.runtime != "multiprocess")
        throw std::runtime_error("runtime must be simulated or multiprocess");
    if (a.threads_per_shard <= 0)
        throw std::runtime_error("threads-per-shard must be positive");
    if (a.index_mode != "build" && a.index_mode != "load" && a.index_mode != "build-if-missing")
        throw std::runtime_error("index-mode must be build, load, or build-if-missing");
    if (a.index_mode != "build" && a.index_dir.empty())
        throw std::runtime_error("--index-dir is required when loading an index");
    if (a.postings_per_shard <= 0 || a.nprobe <= 0 || a.nprobe > a.postings_per_shard || a.kmeans_iterations <= 0 ||
        a.ssd_bytes_per_ns <= 0)
        throw std::runtime_error("invalid SPANN parameters");
    return a;
}

static float distance_of(const Vec &a, const Vec &b, bool ip) {
    float s = 0;
    if (ip)
        for (size_t i = 0; i < a.size(); ++i)
            s -= a[i] * b[i];
    else
        for (size_t i = 0; i < a.size(); ++i) {
            float d = a[i] - b[i];
            s += d * d;
        }
    return s;
}

struct Item {
    float d;
    int id;
};
struct MinCmp {
    bool operator()(const Item &a, const Item &b) const { return a.d > b.d; }
};
struct MaxCmp {
    bool operator()(const Item &a, const Item &b) const { return a.d < b.d; }
};

class HNSW {
    public:
    HNSW(int dim, int M, int efc, uint64_t seed, bool ip = false) : dim_(dim), M_(M), efc_(efc), ip_(ip), rng_(seed) {}

    void add(Vec v, int global_id) {
        int id = static_cast<int>(data_.size());
        int level = random_level();
        data_.push_back(std::move(v));
        global_.push_back(global_id);
        levels_.push_back(level);
        links_.emplace_back(level + 1);
        if (entry_ < 0) {
            entry_ = id;
            max_level_ = level;
            return;
        }
        int ep = entry_;
        for (int l = max_level_; l > level; --l)
            ep = greedy(data_[id], ep, l);
        for (int l = std::min(level, max_level_); l >= 0; --l) {
            auto candidates = search_layer(data_[id], ep, efc_, l);
            if (!candidates.empty())
                ep = candidates.front().id;
            auto selected = select_neighbors(candidates, l == 0 ? 2 * M_ : M_);
            for (int nb : selected) {
                links_[id][l].push_back(nb);
                links_[nb][l].push_back(id);
                prune(nb, l);
            }
        }
        if (level > max_level_) {
            entry_ = id;
            max_level_ = level;
        }
    }

    int size() const { return static_cast<int>(data_.size()); }
    int entry() const { return entry_; }
    int max_level() const { return max_level_; }
    int global_id(int id) const { return global_[id]; }
    const Vec &vec(int id) const { return data_[id]; }
    const std::vector<int> &neighbors(int id, int level) const { return links_[id][level]; }
    float query_distance(const Vec &q, int id) const { return dist(q, data_[id]); }

    int upper_entry(const Vec &q, uint64_t &distance_count) const {
        if (entry_ < 0)
            return -1;
        int cur = entry_;
        float cd = dist(q, data_[cur]);
        ++distance_count;
        for (int level = max_level_; level > 0; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (int nb : links_[cur][level]) {
                    float d = dist(q, data_[nb]);
                    ++distance_count;
                    if (d < cd) {
                        cd = d;
                        cur = nb;
                        changed = true;
                    }
                }
            }
        }
        return cur;
    }

    void save(const std::string &path) const {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out)
            throw std::runtime_error("cannot create index " + path);
        const char magic[8] = {'H', 'N', 'S', 'W', 'I', 'D', 'X', '1'};
        uint32_t dim = dim_, m = M_, efc = efc_, ip = ip_;
        int32_t entry = entry_, max_level = max_level_;
        uint64_t count = data_.size();
        out.write(magic, sizeof(magic));
        out.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
        out.write(reinterpret_cast<const char *>(&m), sizeof(m));
        out.write(reinterpret_cast<const char *>(&efc), sizeof(efc));
        out.write(reinterpret_cast<const char *>(&ip), sizeof(ip));
        out.write(reinterpret_cast<const char *>(&entry), sizeof(entry));
        out.write(reinterpret_cast<const char *>(&max_level), sizeof(max_level));
        out.write(reinterpret_cast<const char *>(&count), sizeof(count));
        for (size_t i = 0; i < data_.size(); ++i) {
            int32_t global_id = global_[i], level = levels_[i];
            out.write(reinterpret_cast<const char *>(&global_id), sizeof(global_id));
            out.write(reinterpret_cast<const char *>(&level), sizeof(level));
            out.write(reinterpret_cast<const char *>(data_[i].data()), dim_ * sizeof(float));
            for (int l = 0; l <= level; ++l) {
                uint32_t degree = links_[i][l].size();
                out.write(reinterpret_cast<const char *>(&degree), sizeof(degree));
                out.write(reinterpret_cast<const char *>(links_[i][l].data()), degree * sizeof(int));
            }
        }
        if (!out)
            throw std::runtime_error("failed while writing index " + path);
    }

    void load(const std::string &path, int expected_dim, bool expected_ip) {
        std::ifstream in(path, std::ios::binary);
        if (!in)
            throw std::runtime_error("cannot open index " + path);
        char magic[8];
        uint32_t dim = 0, m = 0, efc = 0, ip = 0;
        int32_t entry = -1, max_level = -1;
        uint64_t count = 0;
        in.read(magic, sizeof(magic));
        in.read(reinterpret_cast<char *>(&dim), sizeof(dim));
        in.read(reinterpret_cast<char *>(&m), sizeof(m));
        in.read(reinterpret_cast<char *>(&efc), sizeof(efc));
        in.read(reinterpret_cast<char *>(&ip), sizeof(ip));
        in.read(reinterpret_cast<char *>(&entry), sizeof(entry));
        in.read(reinterpret_cast<char *>(&max_level), sizeof(max_level));
        in.read(reinterpret_cast<char *>(&count), sizeof(count));
        const char expected_magic[8] = {'H', 'N', 'S', 'W', 'I', 'D', 'X', '1'};
        if (!in || !std::equal(std::begin(magic), std::end(magic), std::begin(expected_magic)) ||
            dim != static_cast<uint32_t>(expected_dim) || ip != static_cast<uint32_t>(expected_ip))
            throw std::runtime_error("invalid or incompatible index " + path);
        dim_ = dim;
        M_ = m;
        efc_ = efc;
        ip_ = ip;
        entry_ = entry;
        max_level_ = max_level;
        data_.assign(count, Vec(dim_));
        global_.resize(count);
        levels_.resize(count);
        links_.resize(count);
        for (size_t i = 0; i < count; ++i) {
            int32_t global_id = 0, level = 0;
            in.read(reinterpret_cast<char *>(&global_id), sizeof(global_id));
            in.read(reinterpret_cast<char *>(&level), sizeof(level));
            if (level < 0 || level > 64)
                throw std::runtime_error("invalid HNSW level in " + path);
            global_[i] = global_id;
            levels_[i] = level;
            in.read(reinterpret_cast<char *>(data_[i].data()), dim_ * sizeof(float));
            links_[i].resize(level + 1);
            for (int l = 0; l <= level; ++l) {
                uint32_t degree = 0;
                in.read(reinterpret_cast<char *>(&degree), sizeof(degree));
                links_[i][l].resize(degree);
                in.read(reinterpret_cast<char *>(links_[i][l].data()), degree * sizeof(int));
                for (int neighbor : links_[i][l])
                    if (neighbor < 0 || static_cast<uint64_t>(neighbor) >= count)
                        throw std::runtime_error("invalid HNSW edge in " + path);
            }
        }
        if (!in || entry_ < 0 || static_cast<uint64_t>(entry_) >= count)
            throw std::runtime_error("truncated index " + path);
    }

    private:
    int dim_, M_, efc_, entry_ = -1, max_level_ = -1;
    bool ip_;
    std::mt19937_64 rng_;
    std::vector<Vec> data_;
    std::vector<int> global_, levels_;
    std::vector<std::vector<std::vector<int>>> links_;

    float dist(const Vec &a, const Vec &b) const { return distance_of(a, b, ip_); }

    int random_level() {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        return static_cast<int>(-std::log(std::max(u(rng_), 1e-12)) / std::log(static_cast<double>(M_)));
    }
    int greedy(const Vec &q, int ep, int level) const {
        float best = dist(q, data_[ep]);
        bool changed = true;
        while (changed) {
            changed = false;
            for (int nb : links_[ep][level]) {
                float d = dist(q, data_[nb]);
                if (d < best) {
                    best = d;
                    ep = nb;
                    changed = true;
                }
            }
        }
        return ep;
    }
    std::vector<Item> search_layer(const Vec &q, int ep, int ef, int level) const {
        std::priority_queue<Item, std::vector<Item>, MinCmp> candidates;
        std::priority_queue<Item, std::vector<Item>, MaxCmp> best;
        std::unordered_set<int> seen;
        float d = dist(q, data_[ep]);
        candidates.push({d, ep});
        best.push({d, ep});
        seen.insert(ep);
        while (!candidates.empty()) {
            Item c = candidates.top();
            candidates.pop();
            if (best.size() >= static_cast<size_t>(ef) && c.d > best.top().d)
                break;
            for (int nb : links_[c.id][level])
                if (seen.insert(nb).second) {
                    float nd = dist(q, data_[nb]);
                    if (best.size() < static_cast<size_t>(ef) || nd < best.top().d) {
                        candidates.push({nd, nb});
                        best.push({nd, nb});
                        if (best.size() > static_cast<size_t>(ef))
                            best.pop();
                    }
                }
        }
        std::vector<Item> out;
        while (!best.empty()) {
            out.push_back(best.top());
            best.pop();
        }
        std::sort(out.begin(), out.end(), [](auto &a, auto &b) { return a.d < b.d; });
        return out;
    }
    void prune(int id, int level) {
        auto &n = links_[id][level];
        int cap = level == 0 ? 2 * M_ : M_;
        if (n.size() <= static_cast<size_t>(cap))
            return;
        std::vector<Item> candidates;
        candidates.reserve(n.size());
        for (int x : n)
            candidates.push_back({dist(data_[id], data_[x]), x});
        n = select_neighbors(candidates, cap);
    }
    std::vector<int> select_neighbors(std::vector<Item> candidates, int cap) const {
        std::sort(candidates.begin(), candidates.end(), [](auto &a, auto &b) { return a.d < b.d; });
        std::vector<int> selected, rejected;
        selected.reserve(cap);
        for (const auto &c : candidates) {
            bool diverse = true;
            for (int s : selected)
                if (dist(data_[c.id], data_[s]) < c.d) {
                    diverse = false;
                    break;
                }
            (diverse ? selected : rejected).push_back(c.id);
            if (selected.size() == static_cast<size_t>(cap))
                break;
        }
        for (int x : rejected) {
            if (selected.size() == static_cast<size_t>(cap))
                break;
            selected.push_back(x);
        }
        return selected;
    }
};

struct Posting {
    Vec centroid;
    float radius = 0;
    std::vector<int> members;
};

class SpannIndex {
    public:
    SpannIndex(int dim, int posting_count, int iterations, uint64_t seed)
        : dim_(dim), posting_count_(posting_count), iterations_(iterations), rng_(seed) {}

    void add(Vec v, int global_id) {
        data_.push_back(std::move(v));
        global_.push_back(global_id);
    }

    void build(int thread_count = 1) {
        if (data_.empty())
            return;
        int count = std::min<int>(posting_count_, data_.size());
        postings_.assign(count, Posting{Vec(dim_, 0), 0, {}});
        std::vector<int> order(data_.size());
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng_);
        for (int c = 0; c < count; ++c)
            postings_[c].centroid = data_[order[c]];

        std::vector<int> assignment(data_.size());
        int workers = std::max(1, std::min<int>(thread_count, data_.size()));
        for (int iteration = 0; iteration < iterations_; ++iteration) {
            std::vector<std::vector<Vec>> local_sums(workers, std::vector<Vec>(count, Vec(dim_, 0)));
            std::vector<std::vector<int>> local_sizes(workers, std::vector<int>(count, 0));
            std::vector<std::thread> threads;
            for (int worker = 0; worker < workers; ++worker) {
                threads.emplace_back([&, worker] {
                    size_t begin = data_.size() * worker / workers;
                    size_t end = data_.size() * (worker + 1) / workers;
                    for (size_t i = begin; i < end; ++i) {
                        int best = nearest_centroid(data_[i]);
                        assignment[i] = best;
                        ++local_sizes[worker][best];
                        for (int d = 0; d < dim_; ++d)
                            local_sums[worker][best][d] += data_[i][d];
                    }
                });
            }
            for (auto &thread : threads)
                thread.join();
            std::vector<Vec> sums(count, Vec(dim_, 0));
            std::vector<int> sizes(count, 0);
            for (int worker = 0; worker < workers; ++worker)
                for (int c = 0; c < count; ++c) {
                    sizes[c] += local_sizes[worker][c];
                    for (int d = 0; d < dim_; ++d)
                        sums[c][d] += local_sums[worker][c][d];
                }
            for (int c = 0; c < count; ++c) {
                if (sizes[c] == 0) {
                    postings_[c].centroid = data_[order[(c + iteration + 1) % order.size()]];
                    continue;
                }
                for (int d = 0; d < dim_; ++d)
                    postings_[c].centroid[d] = sums[c][d] / sizes[c];
            }
        }

        for (auto &posting : postings_) {
            posting.members.clear();
            posting.radius = 0;
        }
        std::vector<std::thread> assignment_threads;
        for (int worker = 0; worker < workers; ++worker) {
            assignment_threads.emplace_back([&, worker] {
                size_t begin = data_.size() * worker / workers;
                size_t end = data_.size() * (worker + 1) / workers;
                for (size_t i = begin; i < end; ++i)
                    assignment[i] = nearest_centroid(data_[i]);
            });
        }
        for (auto &thread : assignment_threads)
            thread.join();
        for (size_t i = 0; i < data_.size(); ++i) {
            int c = assignment[i];
            postings_[c].members.push_back(static_cast<int>(i));
            postings_[c].radius =
                std::max(postings_[c].radius, std::sqrt(distance_of(data_[i], postings_[c].centroid, false)));
        }
    }

    void save(const std::string &path) const {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out)
            throw std::runtime_error("cannot create index " + path);
        const char magic[8] = {'S', 'P', 'A', 'N', 'I', 'D', 'X', '1'};
        uint32_t dim = dim_, posting_count = static_cast<uint32_t>(postings_.size());
        uint64_t vector_count = data_.size();
        out.write(magic, sizeof(magic));
        out.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
        out.write(reinterpret_cast<const char *>(&posting_count), sizeof(posting_count));
        out.write(reinterpret_cast<const char *>(&vector_count), sizeof(vector_count));
        for (size_t i = 0; i < data_.size(); ++i) {
            int32_t global_id = global_[i];
            out.write(reinterpret_cast<const char *>(&global_id), sizeof(global_id));
            out.write(reinterpret_cast<const char *>(data_[i].data()), dim_ * sizeof(float));
        }
        for (const auto &posting : postings_) {
            uint64_t member_count = posting.members.size();
            out.write(reinterpret_cast<const char *>(posting.centroid.data()), dim_ * sizeof(float));
            out.write(reinterpret_cast<const char *>(&posting.radius), sizeof(posting.radius));
            out.write(reinterpret_cast<const char *>(&member_count), sizeof(member_count));
            out.write(reinterpret_cast<const char *>(posting.members.data()), member_count * sizeof(int));
        }
        if (!out)
            throw std::runtime_error("failed while writing index " + path);
    }

    void load(const std::string &path, int expected_dim) {
        std::ifstream in(path, std::ios::binary);
        if (!in)
            throw std::runtime_error("cannot open index " + path);
        char magic[8];
        uint32_t dim = 0, posting_count = 0;
        uint64_t vector_count = 0;
        in.read(magic, sizeof(magic));
        in.read(reinterpret_cast<char *>(&dim), sizeof(dim));
        in.read(reinterpret_cast<char *>(&posting_count), sizeof(posting_count));
        in.read(reinterpret_cast<char *>(&vector_count), sizeof(vector_count));
        const char expected_magic[8] = {'S', 'P', 'A', 'N', 'I', 'D', 'X', '1'};
        if (!in || !std::equal(std::begin(magic), std::end(magic), std::begin(expected_magic)) ||
            dim != static_cast<uint32_t>(expected_dim) || posting_count != static_cast<uint32_t>(posting_count_))
            throw std::runtime_error("invalid or incompatible index " + path);
        dim_ = dim;
        data_.assign(vector_count, Vec(dim_));
        global_.resize(vector_count);
        for (size_t i = 0; i < data_.size(); ++i) {
            int32_t global_id = 0;
            in.read(reinterpret_cast<char *>(&global_id), sizeof(global_id));
            global_[i] = global_id;
            in.read(reinterpret_cast<char *>(data_[i].data()), dim_ * sizeof(float));
        }
        postings_.assign(posting_count, Posting{Vec(dim_, 0), 0, {}});
        for (auto &posting : postings_) {
            uint64_t member_count = 0;
            in.read(reinterpret_cast<char *>(posting.centroid.data()), dim_ * sizeof(float));
            in.read(reinterpret_cast<char *>(&posting.radius), sizeof(posting.radius));
            in.read(reinterpret_cast<char *>(&member_count), sizeof(member_count));
            posting.members.resize(member_count);
            in.read(reinterpret_cast<char *>(posting.members.data()), member_count * sizeof(int));
            for (int member : posting.members)
                if (member < 0 || static_cast<size_t>(member) >= data_.size())
                    throw std::runtime_error("invalid posting member in " + path);
        }
        if (!in)
            throw std::runtime_error("truncated index " + path);
    }

    int global_id(int local_id) const { return global_[local_id]; }
    const Vec &vec(int local_id) const { return data_[local_id]; }
    const std::vector<Posting> &postings() const { return postings_; }
    int dim() const { return dim_; }
    size_t size() const { return data_.size(); }

    private:
    int dim_, posting_count_, iterations_;
    std::mt19937_64 rng_;
    std::vector<Vec> data_;
    std::vector<int> global_;
    std::vector<Posting> postings_;

    int nearest_centroid(const Vec &v) const {
        int best = 0;
        float best_distance = std::numeric_limits<float>::infinity();
        for (size_t c = 0; c < postings_.size(); ++c) {
            float d = distance_of(v, postings_[c].centroid, false);
            if (d < best_distance) {
                best_distance = d;
                best = static_cast<int>(c);
            }
        }
        return best;
    }
};

struct SearchState {
    const HNSW *index = nullptr;
    const Vec *q = nullptr;
    int ef = 0, k = 0, expansions = 0;
    uint64_t distances = 0, clock = 0;
    float external_tau = std::numeric_limits<float>::infinity();
    bool done = false, threshold_stopped = false;
    std::priority_queue<Item, std::vector<Item>, MinCmp> candidates;
    std::priority_queue<Item, std::vector<Item>, MaxCmp> best;
    std::unordered_set<int> seen;

    void init(uint64_t distance_ns) {
        int ep = index->upper_entry(*q, distances);
        clock += distances * distance_ns;
        if (ep < 0) {
            done = true;
            return;
        }
        float d = index->query_distance(*q, ep);
        ++distances;
        clock += distance_ns;
        candidates.push({d, ep});
        best.push({d, ep});
        seen.insert(ep);
    }
    void step(bool shared, double threshold_scale, uint64_t distance_ns, uint64_t expansion_ns) {
        if (done)
            return;
        if (candidates.empty()) {
            done = true;
            return;
        }
        Item c = candidates.top();
        candidates.pop();
        float local_bound =
            best.size() >= static_cast<size_t>(ef) ? best.top().d : std::numeric_limits<float>::infinity();
        if (c.d > local_bound) {
            done = true;
            return;
        }
        // This is the experimental, approximate HNSW early-termination rule.
        float scaled_tau = external_tau >= 0 ? external_tau * threshold_scale : external_tau / threshold_scale;
        if (shared && std::isfinite(external_tau) && c.d > scaled_tau) {
            done = true;
            threshold_stopped = true;
            return;
        }
        uint64_t before = distances;
        for (int nb : index->neighbors(c.id, 0))
            if (seen.insert(nb).second) {
                float d = index->query_distance(*q, nb);
                ++distances;
                if (best.size() < static_cast<size_t>(ef) || d < best.top().d) {
                    candidates.push({d, nb});
                    best.push({d, nb});
                    if (best.size() > static_cast<size_t>(ef))
                        best.pop();
                }
            }
        ++expansions;
        clock += expansion_ns + (distances - before) * distance_ns;
    }
    std::vector<Item> top(int limit) const {
        auto copy = best;
        std::vector<Item> out;
        while (!copy.empty()) {
            auto x = copy.top();
            copy.pop();
            out.push_back({x.d, index->global_id(x.id)});
        }
        std::sort(out.begin(), out.end(), [](auto &a, auto &b) { return a.d < b.d; });
        if (out.size() > static_cast<size_t>(limit))
            out.resize(limit);
        return out;
    }
};

struct Message {
    uint64_t time;
    int type; // 0 shard snapshot -> coordinator, 1 threshold -> shard
    int shard;
    std::vector<Item> values;
    float tau = std::numeric_limits<float>::infinity();
};
struct MsgCmp {
    bool operator()(const Message &a, const Message &b) const { return a.time > b.time; }
};

struct Result {
    uint64_t distances = 0, latency = 0, messages = 0;
    uint64_t posting_reads = 0, posting_prunes = 0, bytes_read = 0;
    int expansions = 0, threshold_stops = 0;
    std::vector<Item> top;
};

static std::vector<Item> merge_top(const std::vector<std::vector<Item>> &lists, int k) {
    std::vector<Item> all;
    for (const auto &v : lists)
        all.insert(all.end(), v.begin(), v.end());
    std::sort(all.begin(), all.end(), [](auto &a, auto &b) { return a.d < b.d; });
    if (all.size() > static_cast<size_t>(k))
        all.resize(k);
    return all;
}

static Result run_query(const std::vector<HNSW> &indexes, const Vec &q, const Args &a, bool shared) {
    std::vector<SearchState> s(indexes.size());
    for (size_t i = 0; i < s.size(); ++i) {
        s[i].index = &indexes[i];
        s[i].q = &q;
        s[i].ef = a.ef_search;
        s[i].k = a.k;
        s[i].init(a.distance_ns);
    }
    std::priority_queue<Message, std::vector<Message>, MsgCmp> events;
    std::vector<std::vector<Item>> coordinator(indexes.size());
    uint64_t messages = 0;
    auto publish = [&](int shard) {
        events.push({s[shard].clock + a.latency_ns / 2, 0, shard, s[shard].top(a.k), 0});
        ++messages;
    };
    if (shared)
        for (size_t i = 0; i < s.size(); ++i)
            publish(static_cast<int>(i));

    while (true) {
        uint64_t next_work = std::numeric_limits<uint64_t>::max();
        int wi = -1;
        for (size_t i = 0; i < s.size(); ++i)
            if (!s[i].done && s[i].clock < next_work) {
                next_work = s[i].clock;
                wi = static_cast<int>(i);
            }
        uint64_t next_event = events.empty() ? std::numeric_limits<uint64_t>::max() : events.top().time;
        if (!events.empty() && next_event <= next_work) {
            Message m = events.top();
            events.pop();
            if (m.type == 0) {
                coordinator[m.shard] = std::move(m.values);
                // A shard may only benefit from candidates discovered by other shards.
                // Excluding the receiver's own snapshot isolates cross-node sharing from
                // ordinary local early termination (and makes one shard a true no-op).
                for (size_t target = 0; target < s.size(); ++target) {
                    std::vector<std::vector<Item>> others;
                    others.reserve(s.size() - 1);
                    for (size_t source = 0; source < s.size(); ++source)
                        if (source != target)
                            others.push_back(coordinator[source]);
                    auto merged = merge_top(others, a.k);
                    if (merged.size() == static_cast<size_t>(a.k)) {
                        events.push({m.time + a.latency_ns / 2, 1, static_cast<int>(target), {}, merged.back().d});
                        ++messages;
                    }
                }
            } else {
                s[m.shard].external_tau = std::min(s[m.shard].external_tau, m.tau);
            }
            continue;
        }
        if (wi < 0)
            break;
        s[wi].step(shared, a.threshold_scale, a.distance_ns, a.expansion_ns);
        if (shared && (s[wi].done || s[wi].expansions % a.publish_every == 0))
            publish(wi);
    }
    std::vector<std::vector<Item>> lists;
    Result r;
    r.messages = messages;
    for (auto &x : s) {
        r.distances += x.distances;
        r.expansions += x.expansions;
        r.threshold_stops += x.threshold_stopped;
        r.latency = std::max(r.latency, x.clock);
        lists.push_back(x.top(a.k));
    }
    r.top = merge_top(lists, a.k);
    return r;
}

struct SpannSearchState {
    const SpannIndex *index = nullptr;
    const Vec *q = nullptr;
    int k = 0, next = 0;
    uint64_t distances = 0, clock = 0, posting_reads = 0, posting_prunes = 0, bytes_read = 0;
    float external_tau = std::numeric_limits<float>::infinity();
    bool done = false;
    struct Probe {
        float centroid_distance;
        float lower_bound;
        int posting;
    };
    std::vector<Probe> probes;
    std::priority_queue<Item, std::vector<Item>, MaxCmp> best;

    void init(int nprobe, uint64_t distance_ns) {
        for (size_t i = 0; i < index->postings().size(); ++i) {
            const auto &posting = index->postings()[i];
            float squared = distance_of(*q, posting.centroid, false);
            float center_distance = std::sqrt(std::max(0.0f, squared));
            float gap = std::max(0.0f, center_distance - posting.radius);
            probes.push_back({squared, gap * gap, static_cast<int>(i)});
            ++distances;
        }
        clock += distances * distance_ns;
        std::sort(probes.begin(), probes.end(),
                  [](const Probe &a, const Probe &b) { return a.centroid_distance < b.centroid_distance; });
        if (probes.size() > static_cast<size_t>(nprobe))
            probes.resize(nprobe);
        done = probes.empty();
    }

    void step(bool shared, const Args &a) {
        if (done)
            return;
        const Probe &probe = probes[next++];
        if (shared && std::isfinite(external_tau) && probe.lower_bound >= external_tau) {
            ++posting_prunes;
        } else {
            const auto &posting = index->postings()[probe.posting];
            uint64_t bytes =
                posting.members.size() * (static_cast<uint64_t>(index->dim()) * sizeof(float) + sizeof(int));
            bytes_read += bytes;
            ++posting_reads;
            clock += a.io_latency_ns + static_cast<uint64_t>(std::ceil(bytes / a.ssd_bytes_per_ns));
            for (int local_id : posting.members) {
                float d = distance_of(*q, index->vec(local_id), false);
                ++distances;
                if (best.size() < static_cast<size_t>(k) || d < best.top().d) {
                    best.push({d, local_id});
                    if (best.size() > static_cast<size_t>(k))
                        best.pop();
                }
            }
            clock += posting.members.size() * a.distance_ns;
        }
        done = next >= static_cast<int>(probes.size());
    }

    std::vector<Item> top(int limit) const {
        auto copy = best;
        std::vector<Item> out;
        while (!copy.empty()) {
            Item x = copy.top();
            copy.pop();
            out.push_back({x.d, index->global_id(x.id)});
        }
        std::sort(out.begin(), out.end(), [](const Item &a, const Item &b) { return a.d < b.d; });
        if (out.size() > static_cast<size_t>(limit))
            out.resize(limit);
        return out;
    }
};

static Result run_spann_query(const std::vector<SpannIndex> &indexes, const Vec &q, const Args &a, bool shared) {
    std::vector<SpannSearchState> states(indexes.size());
    for (size_t i = 0; i < states.size(); ++i) {
        states[i].index = &indexes[i];
        states[i].q = &q;
        states[i].k = a.k;
        states[i].init(a.nprobe, a.distance_ns);
    }
    std::priority_queue<Message, std::vector<Message>, MsgCmp> events;
    std::vector<std::vector<Item>> coordinator(indexes.size());
    uint64_t messages = 0;
    auto publish = [&](int shard) {
        events.push({states[shard].clock + a.latency_ns / 2, 0, shard, states[shard].top(a.k), 0});
        ++messages;
    };

    while (true) {
        uint64_t next_work = std::numeric_limits<uint64_t>::max();
        int worker = -1;
        for (size_t i = 0; i < states.size(); ++i)
            if (!states[i].done && states[i].clock < next_work) {
                next_work = states[i].clock;
                worker = static_cast<int>(i);
            }
        uint64_t next_event = events.empty() ? std::numeric_limits<uint64_t>::max() : events.top().time;
        if (!events.empty() && next_event <= next_work) {
            Message message = events.top();
            events.pop();
            if (message.type == 0) {
                coordinator[message.shard] = std::move(message.values);
                for (size_t target = 0; target < states.size(); ++target) {
                    std::vector<std::vector<Item>> others;
                    for (size_t source = 0; source < states.size(); ++source)
                        if (source != target)
                            others.push_back(coordinator[source]);
                    auto merged = merge_top(others, a.k);
                    if (merged.size() == static_cast<size_t>(a.k)) {
                        events.push(
                            {message.time + a.latency_ns / 2, 1, static_cast<int>(target), {}, merged.back().d});
                        ++messages;
                    }
                }
            } else {
                states[message.shard].external_tau = std::min(states[message.shard].external_tau, message.tau);
            }
            continue;
        }
        if (worker < 0)
            break;
        states[worker].step(shared, a);
        if (shared)
            publish(worker);
    }

    Result result;
    result.messages = messages;
    std::vector<std::vector<Item>> lists;
    for (const auto &state : states) {
        result.distances += state.distances;
        result.posting_reads += state.posting_reads;
        result.posting_prunes += state.posting_prunes;
        result.bytes_read += state.bytes_read;
        result.expansions += state.posting_reads;
        result.latency = std::max(result.latency, state.clock);
        lists.push_back(state.top(a.k));
    }
    result.top = merge_top(lists, a.k);
    return result;
}

static std::vector<Vec> generate(int count, int dim, int clusters, uint64_t seed,
                                 std::vector<Vec> *centers_out = nullptr) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0, 1), noise(0, 0.35f);
    std::vector<Vec> centers(clusters, Vec(dim));
    for (auto &c : centers)
        for (float &x : c)
            x = normal(rng) * 4;
    std::vector<Vec> out(count, Vec(dim));
    for (int i = 0; i < count; ++i) {
        const auto &c = centers[i % clusters];
        for (int d = 0; d < dim; ++d)
            out[i][d] = c[d] + noise(rng);
    }
    std::shuffle(out.begin(), out.end(), rng);
    if (centers_out)
        *centers_out = std::move(centers);
    return out;
}

static std::vector<Vec> make_queries(int count, const std::vector<Vec> &centers, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> noise(0, 0.35f);
    std::vector<Vec> q(count, Vec(centers[0].size()));
    for (int i = 0; i < count; ++i)
        for (size_t d = 0; d < q[i].size(); ++d)
            q[i][d] = centers[i % centers.size()][d] + noise(rng);
    return q;
}

static std::string infer_type(const std::string &path, const std::string &requested) {
    if (requested != "auto")
        return requested;
    if (path.ends_with(".fbin"))
        return "f32";
    if (path.ends_with(".u8bin"))
        return "u8";
    if (path.ends_with(".i8bin"))
        return "i8";
    throw std::runtime_error("cannot infer data type from " + path + "; use --data-type");
}

template <class T> static std::vector<Vec> load_bin_typed(const std::string &path, uint64_t limit, int &dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    uint32_t count = 0, file_dim = 0;
    in.read(reinterpret_cast<char *>(&count), 4);
    in.read(reinterpret_cast<char *>(&file_dim), 4);
    if (!in || count == 0 || file_dim == 0)
        throw std::runtime_error("invalid binary header in " + path);
    uint64_t take = limit ? std::min<uint64_t>(limit, count) : count;
    dim = static_cast<int>(file_dim);
    std::vector<T> row(file_dim);
    std::vector<Vec> out(take, Vec(file_dim));
    for (uint64_t i = 0; i < take; ++i) {
        in.read(reinterpret_cast<char *>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(T)));
        if (!in)
            throw std::runtime_error("truncated vector payload in " + path);
        for (uint32_t d = 0; d < file_dim; ++d)
            out[i][d] = static_cast<float>(row[d]);
    }
    return out;
}

static std::vector<Vec> load_bin(const std::string &path, const std::string &requested_type, uint64_t limit, int &dim) {
    std::string type = infer_type(path, requested_type);
    if (type == "f32")
        return load_bin_typed<float>(path, limit, dim);
    if (type == "u8")
        return load_bin_typed<uint8_t>(path, limit, dim);
    return load_bin_typed<int8_t>(path, limit, dim);
}

static std::vector<std::vector<Item>> load_groundtruth(const std::string &path, uint64_t query_limit,
                                                       int &requested_k) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    uint32_t nq = 0, file_k = 0;
    in.read(reinterpret_cast<char *>(&nq), 4);
    in.read(reinterpret_cast<char *>(&file_k), 4);
    if (!in || nq == 0 || file_k == 0)
        throw std::runtime_error("invalid k-NN ground truth in " + path);
    if (requested_k > static_cast<int>(file_k)) {
        std::cerr << "warning: requested --k " << requested_k << " exceeds ground-truth K " << file_k
                  << "; clamping k to " << file_k << '\n';
        requested_k = static_cast<int>(file_k);
    }
    uint64_t take = query_limit ? std::min<uint64_t>(query_limit, nq) : nq;
    std::vector<uint32_t> ids(static_cast<size_t>(nq) * file_k);
    in.read(reinterpret_cast<char *>(ids.data()), static_cast<std::streamsize>(ids.size() * sizeof(uint32_t)));
    if (!in)
        throw std::runtime_error("truncated ground-truth IDs in " + path);
    std::vector<float> distances(static_cast<size_t>(nq) * file_k);
    in.read(reinterpret_cast<char *>(distances.data()), static_cast<std::streamsize>(distances.size() * sizeof(float)));
    if (!in)
        throw std::runtime_error("truncated ground-truth distances in " + path);
    std::vector<std::vector<Item>> out(take);
    for (uint64_t i = 0; i < take; ++i) {
        out[i].reserve(requested_k);
        for (int j = 0; j < requested_k; ++j)
            out[i].push_back({distances[i * file_k + j], static_cast<int>(ids[i * file_k + j])});
    }
    return out;
}

static std::vector<Item> exact(const std::vector<Vec> &data, const Vec &q, int k, bool ip) {
    std::vector<Item> x;
    x.reserve(data.size());
    for (size_t i = 0; i < data.size(); ++i)
        x.push_back({distance_of(data[i], q, ip), static_cast<int>(i)});
    std::partial_sort(x.begin(), x.begin() + k, x.end(), [](auto &a, auto &b) { return a.d < b.d; });
    x.resize(k);
    return x;
}

static double recall(const std::vector<Item> &got, const std::vector<Item> &truth) {
    std::unordered_set<int> ids;
    for (auto &x : truth)
        ids.insert(x.id);
    int hit = 0;
    float boundary = truth.back().d;
    float tolerance = 1e-5f * std::max(1.0f, std::abs(boundary));
    for (auto &x : got)
        hit += ids.count(x.id) || std::abs(x.d - boundary) <= tolerance;
    return static_cast<double>(std::min<int>(hit, truth.size())) / truth.size();
}

struct Aggregate {
    long double distances = 0, expansions = 0, latency = 0, recall = 0, messages = 0, stops = 0;
    long double posting_reads = 0, posting_prunes = 0, bytes_read = 0;
};

struct alignas(64) SharedThreshold {
    std::atomic<uint32_t> threshold_bits;
    std::atomic<uint32_t> candidate_count;
};

struct ProcessQueryMetric {
    uint64_t distances = 0, expansions = 0, posting_reads = 0, posting_prunes = 0, bytes_read = 0, wall_ns = 0;
    uint64_t threshold_stops = 0;
    int result_count = 0;
};

static std::vector<int> allowed_cpus() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0)
        throw std::runtime_error("sched_getaffinity failed");
    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &set))
            cpus.push_back(cpu);
    return cpus;
}

static void pin_thread(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
        throw std::runtime_error("pthread_setaffinity_np failed for CPU " + std::to_string(cpu));
}

static Aggregate run_spann_multiprocess(const std::vector<SpannIndex> &indexes, const std::vector<Vec> &queries,
                                        const std::vector<std::vector<Item>> &truth, const Args &a, bool shared) {
    const size_t cells = static_cast<size_t>(a.shards) * queries.size();
    const size_t result_cells = cells * a.k;
    auto *slots = static_cast<SharedThreshold *>(
        mmap(nullptr, cells * sizeof(SharedThreshold), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    auto *metrics = static_cast<ProcessQueryMetric *>(
        mmap(nullptr, cells * sizeof(ProcessQueryMetric), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    auto *results = static_cast<Item *>(
        mmap(nullptr, result_cells * sizeof(Item), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    if (slots == MAP_FAILED || metrics == MAP_FAILED || results == MAP_FAILED)
        throw std::runtime_error("shared mmap allocation failed");
    for (size_t i = 0; i < cells; ++i) {
        new (&slots[i]) SharedThreshold;
        slots[i].threshold_bits.store(std::bit_cast<uint32_t>(std::numeric_limits<float>::infinity()));
        slots[i].candidate_count.store(0);
        metrics[i] = {};
    }

    auto cpus = allowed_cpus();
    if (static_cast<size_t>(a.shards * a.threads_per_shard) > cpus.size())
        throw std::runtime_error("not enough allowed CPUs: use shards*threads-per-shard <= " +
                                 std::to_string(cpus.size()));
    std::vector<pid_t> children;
    for (int shard = 0; shard < a.shards; ++shard) {
        pid_t pid = fork();
        if (pid < 0)
            throw std::runtime_error("fork failed");
        if (pid == 0) {
            std::vector<std::thread> workers;
            for (int thread_id = 0; thread_id < a.threads_per_shard; ++thread_id) {
                workers.emplace_back([&, shard, thread_id] {
                    pin_thread(cpus[shard * a.threads_per_shard + thread_id]);
                    for (size_t query_id = thread_id; query_id < queries.size(); query_id += a.threads_per_shard) {
                        size_t cell = query_id * a.shards + shard;
                        SpannSearchState state;
                        state.index = &indexes[shard];
                        state.q = &queries[query_id];
                        state.k = a.k;
                        state.init(a.nprobe, a.distance_ns);
                        auto begin = std::chrono::steady_clock::now();
                        while (!state.done) {
                            if (shared) {
                                float tau = std::numeric_limits<float>::infinity();
                                for (int other = 0; other < a.shards; ++other) {
                                    if (other == shard)
                                        continue;
                                    auto &other_slot = slots[query_id * a.shards + other];
                                    if (other_slot.candidate_count.load(std::memory_order_acquire) >=
                                        static_cast<uint32_t>(a.k)) {
                                        uint32_t bits = other_slot.threshold_bits.load(std::memory_order_acquire);
                                        tau = std::min(tau, std::bit_cast<float>(bits));
                                    }
                                }
                                state.external_tau = tau;
                            }
                            uint64_t reads_before = state.posting_reads;
                            state.step(shared, a);
                            if (state.posting_reads > reads_before && a.io_latency_ns)
                                std::this_thread::sleep_for(std::chrono::nanoseconds(a.io_latency_ns));
                            auto local = state.top(a.k);
                            if (local.size() == static_cast<size_t>(a.k)) {
                                slots[cell].threshold_bits.store(std::bit_cast<uint32_t>(local.back().d),
                                                                 std::memory_order_release);
                                slots[cell].candidate_count.store(a.k, std::memory_order_release);
                            }
                        }
                        auto elapsed = std::chrono::steady_clock::now() - begin;
                        auto local = state.top(a.k);
                        auto &metric = metrics[cell];
                        metric.distances = state.distances;
                        metric.expansions = state.posting_reads;
                        metric.posting_reads = state.posting_reads;
                        metric.posting_prunes = state.posting_prunes;
                        metric.bytes_read = state.bytes_read;
                        metric.wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
                        metric.result_count = local.size();
                        for (size_t i = 0; i < local.size(); ++i)
                            results[cell * a.k + i] = local[i];
                    }
                });
            }
            for (auto &worker : workers)
                worker.join();
            _exit(0);
        }
        children.push_back(pid);
    }
    for (pid_t child : children) {
        int status = 0;
        waitpid(child, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            throw std::runtime_error("shard process failed");
    }

    Aggregate aggregate;
    for (size_t query_id = 0; query_id < queries.size(); ++query_id) {
        std::vector<std::vector<Item>> shard_results(a.shards);
        uint64_t query_wall = 0;
        for (int shard = 0; shard < a.shards; ++shard) {
            size_t cell = query_id * a.shards + shard;
            const auto &metric = metrics[cell];
            aggregate.distances += metric.distances;
            aggregate.expansions += metric.expansions;
            aggregate.posting_reads += metric.posting_reads;
            aggregate.posting_prunes += metric.posting_prunes;
            aggregate.bytes_read += metric.bytes_read;
            aggregate.stops += metric.threshold_stops;
            query_wall = std::max(query_wall, metric.wall_ns);
            for (int i = 0; i < metric.result_count; ++i)
                shard_results[shard].push_back(results[cell * a.k + i]);
        }
        aggregate.latency += query_wall;
        aggregate.recall += recall(merge_top(shard_results, a.k), truth[query_id]);
    }
    munmap(slots, cells * sizeof(SharedThreshold));
    munmap(metrics, cells * sizeof(ProcessQueryMetric));
    munmap(results, result_cells * sizeof(Item));
    return aggregate;
}

static Aggregate run_hnsw_multiprocess(const std::vector<HNSW> &indexes, const std::vector<Vec> &queries,
                                       const std::vector<std::vector<Item>> &truth, const Args &a, bool shared) {
    const size_t cells = static_cast<size_t>(a.shards) * queries.size();
    const size_t result_cells = cells * a.k;
    auto *slots = static_cast<SharedThreshold *>(
        mmap(nullptr, cells * sizeof(SharedThreshold), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    auto *metrics = static_cast<ProcessQueryMetric *>(
        mmap(nullptr, cells * sizeof(ProcessQueryMetric), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    auto *results = static_cast<Item *>(
        mmap(nullptr, result_cells * sizeof(Item), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    if (slots == MAP_FAILED || metrics == MAP_FAILED || results == MAP_FAILED)
        throw std::runtime_error("shared mmap allocation failed");
    for (size_t i = 0; i < cells; ++i) {
        new (&slots[i]) SharedThreshold;
        slots[i].threshold_bits.store(std::bit_cast<uint32_t>(std::numeric_limits<float>::infinity()));
        slots[i].candidate_count.store(0);
        metrics[i] = {};
    }
    auto cpus = allowed_cpus();
    if (static_cast<size_t>(a.shards * a.threads_per_shard) > cpus.size())
        throw std::runtime_error("not enough allowed CPUs: use shards*threads-per-shard <= " +
                                 std::to_string(cpus.size()));
    std::vector<pid_t> children;
    for (int shard = 0; shard < a.shards; ++shard) {
        pid_t pid = fork();
        if (pid < 0)
            throw std::runtime_error("fork failed");
        if (pid == 0) {
            std::vector<std::thread> workers;
            for (int thread_id = 0; thread_id < a.threads_per_shard; ++thread_id) {
                workers.emplace_back([&, shard, thread_id] {
                    pin_thread(cpus[shard * a.threads_per_shard + thread_id]);
                    for (size_t query_id = thread_id; query_id < queries.size(); query_id += a.threads_per_shard) {
                        size_t cell = query_id * a.shards + shard;
                        SearchState state;
                        state.index = &indexes[shard];
                        state.q = &queries[query_id];
                        state.ef = a.ef_search;
                        state.k = a.k;
                        state.init(a.distance_ns);
                        auto begin = std::chrono::steady_clock::now();
                        while (!state.done) {
                            if (shared) {
                                float tau = std::numeric_limits<float>::infinity();
                                for (int other = 0; other < a.shards; ++other) {
                                    if (other == shard)
                                        continue;
                                    auto &other_slot = slots[query_id * a.shards + other];
                                    if (other_slot.candidate_count.load(std::memory_order_acquire) >=
                                        static_cast<uint32_t>(a.k)) {
                                        uint32_t bits = other_slot.threshold_bits.load(std::memory_order_acquire);
                                        tau = std::min(tau, std::bit_cast<float>(bits));
                                    }
                                }
                                state.external_tau = tau;
                            }
                            state.step(shared, a.threshold_scale, a.distance_ns, a.expansion_ns);
                            auto local = state.top(a.k);
                            if (local.size() == static_cast<size_t>(a.k)) {
                                slots[cell].threshold_bits.store(std::bit_cast<uint32_t>(local.back().d),
                                                                 std::memory_order_release);
                                slots[cell].candidate_count.store(a.k, std::memory_order_release);
                            }
                        }
                        auto elapsed = std::chrono::steady_clock::now() - begin;
                        auto local = state.top(a.k);
                        auto &metric = metrics[cell];
                        metric.distances = state.distances;
                        metric.expansions = state.expansions;
                        metric.threshold_stops = state.threshold_stopped;
                        metric.wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
                        metric.result_count = local.size();
                        for (size_t i = 0; i < local.size(); ++i)
                            results[cell * a.k + i] = local[i];
                    }
                });
            }
            for (auto &worker : workers)
                worker.join();
            _exit(0);
        }
        children.push_back(pid);
    }
    for (pid_t child : children) {
        int status = 0;
        waitpid(child, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            throw std::runtime_error("shard process failed");
    }
    Aggregate aggregate;
    for (size_t query_id = 0; query_id < queries.size(); ++query_id) {
        std::vector<std::vector<Item>> shard_results(a.shards);
        uint64_t query_wall = 0;
        for (int shard = 0; shard < a.shards; ++shard) {
            size_t cell = query_id * a.shards + shard;
            const auto &metric = metrics[cell];
            aggregate.distances += metric.distances;
            aggregate.expansions += metric.expansions;
            aggregate.stops += metric.threshold_stops;
            query_wall = std::max(query_wall, metric.wall_ns);
            for (int i = 0; i < metric.result_count; ++i)
                shard_results[shard].push_back(results[cell * a.k + i]);
        }
        aggregate.latency += query_wall;
        aggregate.recall += recall(merge_top(shard_results, a.k), truth[query_id]);
    }
    munmap(slots, cells * sizeof(SharedThreshold));
    munmap(metrics, cells * sizeof(ProcessQueryMetric));
    munmap(results, result_cells * sizeof(Item));
    return aggregate;
}

int main(int argc, char **argv) try {
    Args a = parse_args(argc, argv);
    auto index_path = [&](int shard) {
        std::string extension = a.engine == "spann" ? ".spann" : ".hnsw";
        return (std::filesystem::path(a.index_dir) / ("shard_" + std::to_string(shard) + extension)).string();
    };
    bool load_persisted = a.index_mode == "load";
    if (a.index_mode == "build-if-missing") {
        load_persisted = true;
        for (int shard = 0; shard < a.shards; ++shard)
            load_persisted = load_persisted && std::filesystem::is_regular_file(index_path(shard));
    }
    if (a.gt_base_size == 0 && a.gt_path.find("10M") != std::string::npos)
        a.gt_base_size = 10000000;
    if (a.gt_base_size != 0) {
        if (a.max_base != 0 && a.max_base != a.gt_base_size)
            throw std::runtime_error("--max-base must match --gt-base-size when ground truth is supplied");
        a.max_base = a.gt_base_size;
    }
    std::vector<Vec> data, queries;
    std::vector<std::vector<Item>> supplied_truth;
    if (load_persisted) {
        if (a.query_path.empty() || a.gt_path.empty())
            throw std::runtime_error("loading a persisted index requires query and ground truth");
        int query_dim = 0;
        queries = load_bin(a.query_path, a.data_type, a.max_queries, query_dim);
        a.dim = query_dim;
        a.queries = static_cast<int>(queries.size());
        supplied_truth = load_groundtruth(a.gt_path, queries.size(), a.k);
    } else if (a.base_path.empty()) {
        std::vector<Vec> centers;
        data = generate(a.n, a.dim, a.clusters, a.seed, &centers);
        queries = make_queries(a.queries, centers, a.seed + 1);
    } else {
        int base_dim = 0, query_dim = 0;
        data = load_bin(a.base_path, a.data_type, a.max_base, base_dim);
        queries = load_bin(a.query_path, a.data_type, a.max_queries, query_dim);
        if (base_dim != query_dim)
            throw std::runtime_error("base/query dimension mismatch");
        a.dim = base_dim;
        a.n = static_cast<int>(data.size());
        a.queries = static_cast<int>(queries.size());
        if (!a.gt_path.empty()) {
            if (a.max_base && a.gt_base_size == 0)
                throw std::runtime_error(
                    "official ground truth is invalid with --max-base; omit GT to recompute exact truth");
            supplied_truth = load_groundtruth(a.gt_path, queries.size(), a.k);
            if (supplied_truth.size() < queries.size())
                throw std::runtime_error("ground truth has fewer queries than query file");
        }
    }
    if (a.ef_search < a.k)
        throw std::runtime_error("ef-search must be >= effective k (after ground-truth clamping)");
    if (!load_persisted && data.size() < static_cast<size_t>(a.k))
        throw std::runtime_error("base contains fewer than k vectors");
    std::vector<std::vector<Item>> all_truth(a.queries);
    for (int qi = 0; qi < a.queries; ++qi)
        all_truth[qi] = supplied_truth.empty() ? exact(data, queries[qi], a.k, a.metric == "ip") : supplied_truth[qi];
    bool ip = a.metric == "ip";
    if (a.engine == "spann" && ip)
        throw std::runtime_error("SPANN radius-bound probe currently supports only L2");
    std::vector<HNSW> hnsw_indexes;
    std::vector<SpannIndex> spann_indexes;
    if (a.engine == "hnsw") {
        hnsw_indexes.reserve(a.shards);
        for (int i = 0; i < a.shards; ++i)
            hnsw_indexes.emplace_back(a.dim, a.M, a.ef_construction, a.seed + 100 + i, ip);
        if (load_persisted) {
            size_t total_vectors = 0;
            for (int shard = 0; shard < a.shards; ++shard) {
                std::cerr << "loading HNSW shard " << shard << " from " << index_path(shard) << '\n';
                hnsw_indexes[shard].load(index_path(shard), a.dim, ip);
                total_vectors += hnsw_indexes[shard].size();
            }
            a.n = static_cast<int>(total_vectors);
            if (a.gt_base_size != 0 && total_vectors != a.gt_base_size)
                throw std::runtime_error("persisted index vector count does not match ground-truth base size");
        } else {
            std::cerr << "building HNSW " << a.shards << " shards over " << a.n << " vectors...\n";
            std::vector<std::vector<Vec>> shard_data(a.shards);
            std::vector<std::vector<int>> shard_ids(a.shards);
            for (int i = 0; i < a.n; ++i) {
                shard_data[i % a.shards].push_back(std::move(data[i]));
                shard_ids[i % a.shards].push_back(i);
            }
            data.clear();
            data.shrink_to_fit();
            std::vector<std::thread> builders;
            for (int shard = 0; shard < a.shards; ++shard) {
                builders.emplace_back([&, shard] {
                    std::cerr << "building HNSW shard " << shard << '\n';
                    for (size_t i = 0; i < shard_data[shard].size(); ++i)
                        hnsw_indexes[shard].add(std::move(shard_data[shard][i]), shard_ids[shard][i]);
                    std::cerr << "finished HNSW shard " << shard << '\n';
                });
            }
            for (auto &builder : builders)
                builder.join();
            if (!a.index_dir.empty()) {
                std::filesystem::create_directories(a.index_dir);
                for (int shard = 0; shard < a.shards; ++shard)
                    hnsw_indexes[shard].save(index_path(shard));
            }
        }
    } else {
        spann_indexes.reserve(a.shards);
        for (int i = 0; i < a.shards; ++i)
            spann_indexes.emplace_back(a.dim, a.postings_per_shard, a.kmeans_iterations, a.seed + 100 + i);
        if (load_persisted) {
            size_t total_vectors = 0;
            for (int shard = 0; shard < a.shards; ++shard) {
                std::cerr << "loading SPANN shard " << shard << " from " << index_path(shard) << '\n';
                spann_indexes[shard].load(index_path(shard), a.dim);
                total_vectors += spann_indexes[shard].size();
            }
            a.n = static_cast<int>(total_vectors);
            if (a.gt_base_size != 0 && total_vectors != a.gt_base_size)
                throw std::runtime_error("persisted index vector count does not match ground-truth base size");
        } else {
            std::cerr << "building SPANN " << a.shards << " shards over " << a.n << " vectors...\n";
            bool can_move_base = true;
            for (int i = 0; i < a.n; ++i) {
                if (can_move_base)
                    spann_indexes[i % a.shards].add(std::move(data[i]), i);
                else
                    spann_indexes[i % a.shards].add(data[i], i);
            }
            if (can_move_base) {
                data.clear();
                data.shrink_to_fit();
            }
            std::vector<std::thread> builders;
            for (int shard = 0; shard < a.shards; ++shard) {
                builders.emplace_back([&, shard] {
                    std::cerr << "building SPANN shard " << shard << " with " << a.threads_per_shard << " threads\n";
                    spann_indexes[shard].build(a.threads_per_shard);
                    std::cerr << "finished SPANN shard " << shard << '\n';
                });
            }
            for (auto &builder : builders)
                builder.join();
            if (!a.index_dir.empty()) {
                std::filesystem::create_directories(a.index_dir);
                for (int shard = 0; shard < a.shards; ++shard) {
                    std::cerr << "saving SPANN shard " << shard << " to " << index_path(shard) << '\n';
                    spann_indexes[shard].save(index_path(shard));
                }
            }
        }
    }

    Aggregate base, shared;
    auto accumulate = [&](Aggregate &aggregate, const Result &result, const std::vector<Item> &truth) {
        aggregate.distances += result.distances;
        aggregate.expansions += result.expansions;
        aggregate.latency += result.latency;
        aggregate.recall += recall(result.top, truth);
        aggregate.messages += result.messages;
        aggregate.stops += result.threshold_stops;
        aggregate.posting_reads += result.posting_reads;
        aggregate.posting_prunes += result.posting_prunes;
        aggregate.bytes_read += result.bytes_read;
    };
    if (a.runtime == "multiprocess") {
        if (a.mode != "shared")
            base = a.engine == "spann" ? run_spann_multiprocess(spann_indexes, queries, all_truth, a, false)
                                       : run_hnsw_multiprocess(hnsw_indexes, queries, all_truth, a, false);
        if (a.mode != "independent")
            shared = a.engine == "spann" ? run_spann_multiprocess(spann_indexes, queries, all_truth, a, true)
                                         : run_hnsw_multiprocess(hnsw_indexes, queries, all_truth, a, true);
    } else {
        for (int qi = 0; qi < a.queries; ++qi) {
            if (a.mode != "shared") {
                auto result = a.engine == "hnsw" ? run_query(hnsw_indexes, queries[qi], a, false)
                                                 : run_spann_query(spann_indexes, queries[qi], a, false);
                accumulate(base, result, all_truth[qi]);
            }
            if (a.mode != "independent") {
                auto result = a.engine == "hnsw" ? run_query(hnsw_indexes, queries[qi], a, true)
                                                 : run_spann_query(spann_indexes, queries[qi], a, true);
                accumulate(shared, result, all_truth[qi]);
            }
        }
    }
    std::cout << "mode,engine,n,queries,dim,shards,k,ef_search,latency_ns,threshold_scale,avg_distance_computations,"
                 "avg_expansions,avg_latency_ns,recall,avg_messages,avg_threshold_stops,avg_posting_reads,"
                 "avg_posting_prunes,avg_bytes_read\n";
    auto print = [&](const char *name, const Aggregate &x) {
        long double q = a.queries;
        std::cout << name << ',' << a.engine << ',' << a.n << ',' << a.queries << ',' << a.dim << ',' << a.shards << ','
                  << a.k << ',' << a.ef_search << ',' << a.latency_ns << ',' << a.threshold_scale << ',' << std::fixed
                  << std::setprecision(3) << static_cast<double>(x.distances / q) << ','
                  << static_cast<double>(x.expansions / q) << ',' << static_cast<double>(x.latency / q) << ','
                  << static_cast<double>(x.recall / q) << ',' << static_cast<double>(x.messages / q) << ','
                  << static_cast<double>(x.stops / q) << ',' << static_cast<double>(x.posting_reads / q) << ','
                  << static_cast<double>(x.posting_prunes / q) << ',' << static_cast<double>(x.bytes_read / q) << '\n';
    };
    if (a.mode != "shared")
        print("independent", base);
    if (a.mode != "independent")
        print("shared", shared);
    return 0;
} catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
