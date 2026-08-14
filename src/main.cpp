#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using Vec = std::vector<float>;

struct Args {
  int n = 20000, queries = 200, dim = 32, shards = 4, k = 10;
  int M = 16, ef_construction = 100, ef_search = 80, clusters = 32;
  int publish_every = 1;
  uint64_t seed = 42, latency_ns = 0, distance_ns = 50, expansion_ns = 100;
  double threshold_scale = 1.0;
  std::string mode = "all";
  std::string base_path, query_path, gt_path, data_type = "auto", metric = "l2";
  uint64_t max_base = 0, max_queries = 0;
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
            << "  --max-base N --max-queries N\n";
}

static Args parse_args(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--help" || key == "-h") { usage(argv[0]); std::exit(0); }
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
    std::string v = argv[++i];
    auto iv = [&] { return std::stoi(v); };
    auto uv = [&] { return std::stoull(v); };
    if (key == "--n") a.n = iv(); else if (key == "--queries") a.queries = iv();
    else if (key == "--dim") a.dim = iv(); else if (key == "--shards") a.shards = iv();
    else if (key == "--k") a.k = iv(); else if (key == "--M") a.M = iv();
    else if (key == "--ef-construction") a.ef_construction = iv();
    else if (key == "--ef-search") a.ef_search = iv(); else if (key == "--clusters") a.clusters = iv();
    else if (key == "--publish-every") a.publish_every = iv(); else if (key == "--seed") a.seed = uv();
    else if (key == "--latency-ns") a.latency_ns = uv(); else if (key == "--distance-ns") a.distance_ns = uv();
    else if (key == "--expansion-ns") a.expansion_ns = uv(); else if (key == "--mode") a.mode = v;
    else if (key == "--threshold-scale") a.threshold_scale = std::stod(v);
    else if (key == "--base") a.base_path = v; else if (key == "--query") a.query_path = v;
    else if (key == "--groundtruth") a.gt_path = v; else if (key == "--data-type") a.data_type = v;
    else if (key == "--metric") a.metric = v; else if (key == "--max-base") a.max_base = uv();
    else if (key == "--max-queries") a.max_queries = uv();
    else throw std::runtime_error("unknown option: " + key);
  }
  if (a.n <= 0 || a.queries <= 0 || a.dim <= 0 || a.shards <= 0 || a.k <= 0 || a.M <= 1 ||
      a.ef_search < a.k || a.ef_construction < a.M || a.publish_every <= 0 || a.clusters <= 0 || a.threshold_scale < 1.0)
    throw std::runtime_error("invalid arguments (ef-search must be >= k)");
  if (a.mode != "all" && a.mode != "independent" && a.mode != "shared")
    throw std::runtime_error("mode must be all, independent, or shared");
  if (a.metric != "l2" && a.metric != "ip") throw std::runtime_error("metric must be l2 or ip");
  if (a.data_type != "auto" && a.data_type != "f32" && a.data_type != "u8" && a.data_type != "i8")
    throw std::runtime_error("data-type must be auto, f32, u8, or i8");
  if (a.base_path.empty() != a.query_path.empty()) throw std::runtime_error("--base and --query must be provided together");
  return a;
}

static float distance_of(const Vec &a, const Vec &b, bool ip) {
  float s = 0;
  if (ip) for (size_t i = 0; i < a.size(); ++i) s -= a[i] * b[i];
  else for (size_t i = 0; i < a.size(); ++i) { float d = a[i] - b[i]; s += d * d; }
  return s;
}

struct Item { float d; int id; };
struct MinCmp { bool operator()(const Item &a, const Item &b) const { return a.d > b.d; } };
struct MaxCmp { bool operator()(const Item &a, const Item &b) const { return a.d < b.d; } };

class HNSW {
 public:
  HNSW(int dim, int M, int efc, uint64_t seed, bool ip=false) : dim_(dim), M_(M), efc_(efc), ip_(ip), rng_(seed) {}

  void add(Vec v, int global_id) {
    int id = static_cast<int>(data_.size());
    int level = random_level();
    data_.push_back(std::move(v)); global_.push_back(global_id); levels_.push_back(level);
    links_.emplace_back(level + 1);
    if (entry_ < 0) { entry_ = id; max_level_ = level; return; }
    int ep = entry_;
    for (int l = max_level_; l > level; --l) ep = greedy(data_[id], ep, l);
    for (int l = std::min(level, max_level_); l >= 0; --l) {
      auto candidates = search_layer(data_[id], ep, efc_, l);
      if (!candidates.empty()) ep = candidates.front().id;
      auto selected = select_neighbors(candidates, l == 0 ? 2 * M_ : M_);
      for (int nb : selected) {
        links_[id][l].push_back(nb);
        links_[nb][l].push_back(id);
        prune(nb, l);
      }
    }
    if (level > max_level_) { entry_ = id; max_level_ = level; }
  }

  int size() const { return static_cast<int>(data_.size()); }
  int entry() const { return entry_; }
  int max_level() const { return max_level_; }
  int global_id(int id) const { return global_[id]; }
  const Vec &vec(int id) const { return data_[id]; }
  const std::vector<int> &neighbors(int id, int level) const { return links_[id][level]; }
  float query_distance(const Vec &q, int id) const { return dist(q, data_[id]); }

  int upper_entry(const Vec &q, uint64_t &distance_count) const {
    if (entry_ < 0) return -1;
    int cur = entry_; float cd = dist(q, data_[cur]); ++distance_count;
    for (int level = max_level_; level > 0; --level) {
      bool changed = true;
      while (changed) {
        changed = false;
        for (int nb : links_[cur][level]) {
          float d = dist(q, data_[nb]); ++distance_count;
          if (d < cd) { cd = d; cur = nb; changed = true; }
        }
      }
    }
    return cur;
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
    float best = dist(q, data_[ep]); bool changed = true;
    while (changed) {
      changed = false;
      for (int nb : links_[ep][level]) { float d = dist(q, data_[nb]); if (d < best) { best = d; ep = nb; changed = true; } }
    }
    return ep;
  }
  std::vector<Item> search_layer(const Vec &q, int ep, int ef, int level) const {
    std::priority_queue<Item, std::vector<Item>, MinCmp> candidates;
    std::priority_queue<Item, std::vector<Item>, MaxCmp> best;
    std::unordered_set<int> seen;
    float d = dist(q, data_[ep]); candidates.push({d, ep}); best.push({d, ep}); seen.insert(ep);
    while (!candidates.empty()) {
      Item c = candidates.top(); candidates.pop();
      if (best.size() >= static_cast<size_t>(ef) && c.d > best.top().d) break;
      for (int nb : links_[c.id][level]) if (seen.insert(nb).second) {
        float nd = dist(q, data_[nb]);
        if (best.size() < static_cast<size_t>(ef) || nd < best.top().d) {
          candidates.push({nd, nb}); best.push({nd, nb});
          if (best.size() > static_cast<size_t>(ef)) best.pop();
        }
      }
    }
    std::vector<Item> out;
    while (!best.empty()) { out.push_back(best.top()); best.pop(); }
    std::sort(out.begin(), out.end(), [](auto &a, auto &b) { return a.d < b.d; });
    return out;
  }
  void prune(int id, int level) {
    auto &n = links_[id][level];
    int cap = level == 0 ? 2 * M_ : M_;
    if (n.size() <= static_cast<size_t>(cap)) return;
    std::vector<Item> candidates; candidates.reserve(n.size());
    for (int x : n) candidates.push_back({dist(data_[id], data_[x]), x});
    n = select_neighbors(candidates, cap);
  }
  std::vector<int> select_neighbors(std::vector<Item> candidates, int cap) const {
    std::sort(candidates.begin(), candidates.end(), [](auto &a, auto &b) { return a.d < b.d; });
    std::vector<int> selected, rejected; selected.reserve(cap);
    for (const auto &c : candidates) {
      bool diverse = true;
      for (int s : selected) if (dist(data_[c.id], data_[s]) < c.d) { diverse = false; break; }
      (diverse ? selected : rejected).push_back(c.id);
      if (selected.size() == static_cast<size_t>(cap)) break;
    }
    for (int x : rejected) {
      if (selected.size() == static_cast<size_t>(cap)) break;
      selected.push_back(x);
    }
    return selected;
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
    if (ep < 0) { done = true; return; }
    float d = index->query_distance(*q, ep); ++distances; clock += distance_ns;
    candidates.push({d, ep}); best.push({d, ep}); seen.insert(ep);
  }
  void step(bool shared, double threshold_scale, uint64_t distance_ns, uint64_t expansion_ns) {
    if (done) return;
    if (candidates.empty()) { done = true; return; }
    Item c = candidates.top(); candidates.pop();
    float local_bound = best.size() >= static_cast<size_t>(ef) ? best.top().d : std::numeric_limits<float>::infinity();
    if (c.d > local_bound) { done = true; return; }
    // This is the experimental, approximate HNSW early-termination rule.
    float scaled_tau = external_tau >= 0 ? external_tau * threshold_scale : external_tau / threshold_scale;
    if (shared && std::isfinite(external_tau) && c.d > scaled_tau) {
      done = true; threshold_stopped = true; return;
    }
    uint64_t before = distances;
    for (int nb : index->neighbors(c.id, 0)) if (seen.insert(nb).second) {
      float d = index->query_distance(*q, nb); ++distances;
      if (best.size() < static_cast<size_t>(ef) || d < best.top().d) {
        candidates.push({d, nb}); best.push({d, nb});
        if (best.size() > static_cast<size_t>(ef)) best.pop();
      }
    }
    ++expansions;
    clock += expansion_ns + (distances - before) * distance_ns;
  }
  std::vector<Item> top(int limit) const {
    auto copy = best; std::vector<Item> out;
    while (!copy.empty()) { auto x = copy.top(); copy.pop(); out.push_back({x.d, index->global_id(x.id)}); }
    std::sort(out.begin(), out.end(), [](auto &a, auto &b) { return a.d < b.d; });
    if (out.size() > static_cast<size_t>(limit)) out.resize(limit);
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
struct MsgCmp { bool operator()(const Message &a, const Message &b) const { return a.time > b.time; } };

struct Result {
  uint64_t distances = 0, latency = 0, messages = 0;
  int expansions = 0, threshold_stops = 0;
  std::vector<Item> top;
};

static std::vector<Item> merge_top(const std::vector<std::vector<Item>> &lists, int k) {
  std::vector<Item> all;
  for (const auto &v : lists) all.insert(all.end(), v.begin(), v.end());
  std::sort(all.begin(), all.end(), [](auto &a, auto &b) { return a.d < b.d; });
  if (all.size() > static_cast<size_t>(k)) all.resize(k);
  return all;
}

static Result run_query(const std::vector<HNSW> &indexes, const Vec &q, const Args &a, bool shared) {
  std::vector<SearchState> s(indexes.size());
  for (size_t i = 0; i < s.size(); ++i) { s[i].index = &indexes[i]; s[i].q = &q; s[i].ef = a.ef_search; s[i].k = a.k; s[i].init(a.distance_ns); }
  std::priority_queue<Message, std::vector<Message>, MsgCmp> events;
  std::vector<std::vector<Item>> coordinator(indexes.size());
  uint64_t messages = 0;
  auto publish = [&](int shard) {
    events.push({s[shard].clock + a.latency_ns / 2, 0, shard, s[shard].top(a.k), 0}); ++messages;
  };
  if (shared) for (size_t i = 0; i < s.size(); ++i) publish(static_cast<int>(i));

  while (true) {
    uint64_t next_work = std::numeric_limits<uint64_t>::max(); int wi = -1;
    for (size_t i = 0; i < s.size(); ++i) if (!s[i].done && s[i].clock < next_work) { next_work = s[i].clock; wi = static_cast<int>(i); }
    uint64_t next_event = events.empty() ? std::numeric_limits<uint64_t>::max() : events.top().time;
    if (!events.empty() && next_event <= next_work) {
      Message m = events.top(); events.pop();
      if (m.type == 0) {
        coordinator[m.shard] = std::move(m.values);
        auto merged = merge_top(coordinator, a.k);
        if (merged.size() == static_cast<size_t>(a.k)) {
          float tau = merged.back().d;
          for (size_t i = 0; i < s.size(); ++i) { events.push({m.time + a.latency_ns / 2, 1, static_cast<int>(i), {}, tau}); ++messages; }
        }
      } else {
        s[m.shard].external_tau = std::min(s[m.shard].external_tau, m.tau);
      }
      continue;
    }
    if (wi < 0) break;
    s[wi].step(shared, a.threshold_scale, a.distance_ns, a.expansion_ns);
    if (shared && (s[wi].done || s[wi].expansions % a.publish_every == 0)) publish(wi);
  }
  std::vector<std::vector<Item>> lists;
  Result r; r.messages = messages;
  for (auto &x : s) {
    r.distances += x.distances; r.expansions += x.expansions; r.threshold_stops += x.threshold_stopped;
    r.latency = std::max(r.latency, x.clock); lists.push_back(x.top(a.k));
  }
  r.top = merge_top(lists, a.k);
  return r;
}

static std::vector<Vec> generate(int count, int dim, int clusters, uint64_t seed, std::vector<Vec> *centers_out = nullptr) {
  std::mt19937_64 rng(seed); std::normal_distribution<float> normal(0, 1), noise(0, 0.35f);
  std::vector<Vec> centers(clusters, Vec(dim));
  for (auto &c : centers) for (float &x : c) x = normal(rng) * 4;
  std::vector<Vec> out(count, Vec(dim));
  for (int i = 0; i < count; ++i) { const auto &c = centers[i % clusters]; for (int d = 0; d < dim; ++d) out[i][d] = c[d] + noise(rng); }
  std::shuffle(out.begin(), out.end(), rng);
  if (centers_out) *centers_out = std::move(centers);
  return out;
}

static std::vector<Vec> make_queries(int count, const std::vector<Vec> &centers, uint64_t seed) {
  std::mt19937_64 rng(seed); std::normal_distribution<float> noise(0, 0.35f);
  std::vector<Vec> q(count, Vec(centers[0].size()));
  for (int i = 0; i < count; ++i) for (size_t d = 0; d < q[i].size(); ++d) q[i][d] = centers[i % centers.size()][d] + noise(rng);
  return q;
}

static std::string infer_type(const std::string &path, const std::string &requested) {
  if (requested != "auto") return requested;
  if (path.ends_with(".fbin")) return "f32";
  if (path.ends_with(".u8bin")) return "u8";
  if (path.ends_with(".i8bin")) return "i8";
  throw std::runtime_error("cannot infer data type from " + path + "; use --data-type");
}

template<class T>
static std::vector<Vec> load_bin_typed(const std::string &path, uint64_t limit, int &dim) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  uint32_t count = 0, file_dim = 0;
  in.read(reinterpret_cast<char *>(&count), 4); in.read(reinterpret_cast<char *>(&file_dim), 4);
  if (!in || count == 0 || file_dim == 0) throw std::runtime_error("invalid binary header in " + path);
  uint64_t take = limit ? std::min<uint64_t>(limit, count) : count;
  dim = static_cast<int>(file_dim);
  std::vector<T> row(file_dim); std::vector<Vec> out(take, Vec(file_dim));
  for (uint64_t i = 0; i < take; ++i) {
    in.read(reinterpret_cast<char *>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(T)));
    if (!in) throw std::runtime_error("truncated vector payload in " + path);
    for (uint32_t d = 0; d < file_dim; ++d) out[i][d] = static_cast<float>(row[d]);
  }
  return out;
}

static std::vector<Vec> load_bin(const std::string &path, const std::string &requested_type, uint64_t limit, int &dim) {
  std::string type = infer_type(path, requested_type);
  if (type == "f32") return load_bin_typed<float>(path, limit, dim);
  if (type == "u8") return load_bin_typed<uint8_t>(path, limit, dim);
  return load_bin_typed<int8_t>(path, limit, dim);
}

static std::vector<std::vector<Item>> load_groundtruth(const std::string &path, uint64_t query_limit, int requested_k) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  uint32_t nq = 0, file_k = 0;
  in.read(reinterpret_cast<char *>(&nq), 4); in.read(reinterpret_cast<char *>(&file_k), 4);
  if (!in || nq == 0 || file_k == 0 || requested_k > static_cast<int>(file_k))
    throw std::runtime_error("invalid or insufficient k-NN ground truth in " + path);
  uint64_t take = query_limit ? std::min<uint64_t>(query_limit, nq) : nq;
  std::vector<uint32_t> ids(static_cast<size_t>(nq) * file_k);
  in.read(reinterpret_cast<char *>(ids.data()), static_cast<std::streamsize>(ids.size() * sizeof(uint32_t)));
  if (!in) throw std::runtime_error("truncated ground-truth IDs in " + path);
  std::vector<float> distances(static_cast<size_t>(nq) * file_k);
  in.read(reinterpret_cast<char *>(distances.data()), static_cast<std::streamsize>(distances.size() * sizeof(float)));
  if (!in) throw std::runtime_error("truncated ground-truth distances in " + path);
  std::vector<std::vector<Item>> out(take);
  for (uint64_t i = 0; i < take; ++i) {
    out[i].reserve(requested_k);
    for (int j = 0; j < requested_k; ++j)
      out[i].push_back({distances[i * file_k + j], static_cast<int>(ids[i * file_k + j])});
  }
  return out;
}

static std::vector<Item> exact(const std::vector<Vec> &data, const Vec &q, int k, bool ip) {
  std::vector<Item> x; x.reserve(data.size());
  for (size_t i = 0; i < data.size(); ++i) x.push_back({distance_of(data[i], q, ip), static_cast<int>(i)});
  std::partial_sort(x.begin(), x.begin() + k, x.end(), [](auto &a, auto &b) { return a.d < b.d; }); x.resize(k); return x;
}

static double recall(const std::vector<Item> &got, const std::vector<Item> &truth) {
  std::unordered_set<int> ids; for (auto &x : truth) ids.insert(x.id);
  int hit = 0;
  float boundary = truth.back().d;
  float tolerance = 1e-5f * std::max(1.0f, std::abs(boundary));
  for (auto &x : got) hit += ids.count(x.id) || std::abs(x.d - boundary) <= tolerance;
  return static_cast<double>(std::min<int>(hit, truth.size())) / truth.size();
}

struct Aggregate { long double distances=0, expansions=0, latency=0, recall=0, messages=0, stops=0; };

int main(int argc, char **argv) try {
  Args a = parse_args(argc, argv);
  std::vector<Vec> data, queries;
  std::vector<std::vector<Item>> supplied_truth;
  if (a.base_path.empty()) {
    std::vector<Vec> centers; data = generate(a.n, a.dim, a.clusters, a.seed, &centers);
    queries = make_queries(a.queries, centers, a.seed + 1);
  } else {
    int base_dim = 0, query_dim = 0;
    data = load_bin(a.base_path, a.data_type, a.max_base, base_dim);
    queries = load_bin(a.query_path, a.data_type, a.max_queries, query_dim);
    if (base_dim != query_dim) throw std::runtime_error("base/query dimension mismatch");
    a.dim = base_dim; a.n = static_cast<int>(data.size()); a.queries = static_cast<int>(queries.size());
    if (!a.gt_path.empty()) {
      if (a.max_base) throw std::runtime_error("official ground truth is invalid with --max-base; omit GT to recompute exact truth");
      supplied_truth = load_groundtruth(a.gt_path, queries.size(), a.k);
      if (supplied_truth.size() < queries.size()) throw std::runtime_error("ground truth has fewer queries than query file");
    }
  }
  if (data.size() < static_cast<size_t>(a.k)) throw std::runtime_error("base contains fewer than k vectors");
  bool ip = a.metric == "ip";
  std::vector<HNSW> indexes; indexes.reserve(a.shards);
  for (int i = 0; i < a.shards; ++i) indexes.emplace_back(a.dim, a.M, a.ef_construction, a.seed + 100 + i, ip);
  std::cerr << "building " << a.shards << " shards over " << a.n << " vectors...\n";
  for (int i = 0; i < a.n; ++i) indexes[i % a.shards].add(data[i], i);

  Aggregate base, shared;
  for (int qi = 0; qi < a.queries; ++qi) {
    auto truth = supplied_truth.empty() ? exact(data, queries[qi], a.k, ip) : supplied_truth[qi];
    if (a.mode != "shared") {
      auto r = run_query(indexes, queries[qi], a, false);
      base.distances += r.distances; base.expansions += r.expansions; base.latency += r.latency;
      base.recall += recall(r.top, truth); base.messages += r.messages; base.stops += r.threshold_stops;
    }
    if (a.mode != "independent") {
      auto r = run_query(indexes, queries[qi], a, true);
      shared.distances += r.distances; shared.expansions += r.expansions; shared.latency += r.latency;
      shared.recall += recall(r.top, truth); shared.messages += r.messages; shared.stops += r.threshold_stops;
    }
  }
  std::cout << "mode,n,queries,dim,shards,k,ef_search,latency_ns,threshold_scale,avg_distance_computations,avg_expansions,avg_simulated_latency_ns,recall,avg_messages,avg_threshold_stops\n";
  auto print = [&](const char *name, const Aggregate &x) {
    long double q = a.queries;
    std::cout << name << ',' << a.n << ',' << a.queries << ',' << a.dim << ',' << a.shards << ',' << a.k << ','
              << a.ef_search << ',' << a.latency_ns << ',' << a.threshold_scale << ',' << std::fixed << std::setprecision(3)
              << static_cast<double>(x.distances/q) << ',' << static_cast<double>(x.expansions/q) << ','
              << static_cast<double>(x.latency/q) << ',' << static_cast<double>(x.recall/q) << ','
              << static_cast<double>(x.messages/q) << ',' << static_cast<double>(x.stops/q) << '\n';
  };
  if (a.mode != "shared") print("independent", base);
  if (a.mode != "independent") print("shared", shared);
  return 0;
} catch (const std::exception &e) {
  std::cerr << "error: " << e.what() << '\n'; return 2;
}
