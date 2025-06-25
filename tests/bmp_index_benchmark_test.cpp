#include "bmp_alg.hpp"
#include "linscan_alg.hpp"
#include "sparse_test_util.hpp"
#include "sparse_util.hpp"
#include <gtest/gtest.h>
#include <stdio.h>
#include <sstream>
#include <chrono>
#include <unordered_set>

using namespace sparse_vector_bmp;

Tuple<u32, u32, UniquePtr<i32[]>, UniquePtr<f32[]>> DecodeGroundtruth(const String &path, bool meta=false) {
    FileHandle file_handle(path);
    file_handle.Open();

    u32 top_k = 0;
    u32 query_n = 0;
    file_handle.Read(&query_n, sizeof(query_n));
    file_handle.Read(&top_k, sizeof(top_k));
    if (meta) {
        return {top_k, query_n, nullptr, nullptr};
    }

    auto indices = MakeUnique<i32[]>(query_n * top_k);
    file_handle.Read(indices.get(), sizeof(i32) * query_n * top_k);
    auto scores = MakeUnique<f32[]>(query_n * top_k);
    file_handle.Read(scores.get(), sizeof(f32) * query_n * top_k);
    return {top_k, query_n, std::move(indices), std::move(scores)};
}

f32 CheckGroundtruth(i32 *gt_indices_list, f32 *gt_score_list, const Vector<Pair<Vector<u32>, Vector<f32>>> &results, u32 top_k) {
    u32 query_n = results.size();

    SizeT recall_n = 0;
    for (u32 i = 0; i < results.size(); ++i) {
        const auto &[indices, scores] = results[i];
        const i32 *gt_indices = gt_indices_list + i * top_k;

        HashSet<u32> indices_set(indices.begin(), indices.end());
        for (u32 j = 0; j < top_k; ++j) {
            if (indices_set.contains(gt_indices[j])) {
                ++recall_n;
            }
        }
    }
    f32 recall = static_cast<f32>(recall_n) / (query_n * top_k);
    return recall;
}

SparseMatrix<f32, i32> DecodeSparseDataset(const String &data_path) {
  FileHandle fh{data_path};
  if(!fh.Open()) {
    GTEST_LOG_(INFO)<< "Open file failed\n";
    return {};
  }
  return SparseMatrix<f32, i32>::Load(fh);
}


const int kQueryLogInterval = 1000;
Vector<Pair<Vector<u32>, Vector<f32>>> Search(i32 thread_n,
                                              const SparseMatrix<f32, i32> &query_mat,
                                              u32 top_k,
                                              i64 query_n,
                                              BmpSearchOptions &opt,
                                              std::function<Pair<Vector<u32>, Vector<f32>>(const SparseVecRef<f32, i32> &, u32, BmpSearchOptions &)> search_fn) {
  Vector<Pair<Vector<u32>, Vector<f32>>> res(query_n);
  Atomic<i64> query_idx{0};
  Vector<Thread> threads;
  for (i32 thread_id = 0; thread_id < thread_n; ++thread_id) {
    threads.emplace_back([&]() {
      while (true) {
        i64 query_i = query_idx.fetch_add(1);
        if (query_i >= query_n) {
            break;
        }
        SparseVecRef<f32, i32> query = query_mat.at(query_i);
        auto [indices, scores] = search_fn(query, top_k, opt);
        res[query_i] = {std::move(indices), std::move(scores)};

        if (kQueryLogInterval != 0 && query_i % kQueryLogInterval == 0) {
            GTEST_LOG_(INFO) << "Querying doc "<< query_i << "\n";
        }
      }
    });
  };
  for (auto &thread : threads) {
      thread.join();
  }
  return res;
}

TEST(BMPTest, benchmark) {

  int block_size = 16;

  SparseMatrix<f32, i32> data_mat = DecodeSparseDataset("/home/qiyu.zd/sparse/data/base_1M.csr");
  SparseMatrix<f32, i32> query_mat = DecodeSparseDataset("/home/qiyu.zd/sparse/data/queries.dev.csr");


  ASSERT_NE(query_mat.nrow_, 0);
  ASSERT_NE(data_mat.nrow_, 0);

  BMPAlg<f32, i16> index(data_mat.ncol_, block_size);
  i64 LogInterval = 100000;
  GTEST_LOG_(INFO) << "Start importing data, total doc count " << data_mat.nrow_;
  
  for (SparseMatrixIter<f32, i32> iter(data_mat); iter.HasNext(); iter.Next()) {
    SparseVecRef<f32, i32> vec = iter.val();
    u32 doc_id = iter.row_id();
    Vector<i16> indices(vec.nnz_);
    for (i32 i = 0; i < vec.nnz_; i++) {
        indices[i] = static_cast<i16>(vec.indices_[i]);
    }
    SparseVecRef<f32, i16> vec1(vec.nnz_, indices.data(), vec.data_);
    index.AddDoc(vec1, doc_id);

    if (LogInterval != 0 && doc_id % LogInterval == 0) {
        GTEST_LOG_(INFO)<<"Imported "<< doc_id << " docs";
        
    }
  }
  GTEST_LOG_(INFO) << "Finish importing data"/*<< dims_avg/data_mat.nrow_*/;
  data_mat.Clear();

  std::chrono::time_point<std::chrono::high_resolution_clock> begin_ts, end_ts;

  Vector<Pair<Vector<u32>, Vector<f32>>> query_result;
  i32 thread_n = 16;
  u32 top_k = 10;
  BmpSearchOptions opt;
  opt.use_lock_ = false;
  
  BMPOptimizeOptions optimize_opt{top_k, true};
  GTEST_LOG_(INFO) << "Start optimizing index";
  index.Optimize(optimize_opt);
  GTEST_LOG_(INFO) << "Finish optimizing index";

  GTEST_LOG_(INFO) << "Start querying with "<<thread_n<<" thread, limit "<<top_k;
  begin_ts = std::chrono::high_resolution_clock::now();

  query_result = Search(thread_n,
          query_mat,
          top_k,
          query_mat.nrow_,
          opt,
          [&](const SparseVecRef<f32, i32> &query, u32 topk, BmpSearchOptions &opt) -> Pair<Vector<u32>, Vector<f32>> {
              Vector<i16> indices(query.nnz_);
              for (i32 i = 0; i < query.nnz_; i++) {
                  indices[i] = static_cast<i16>(query.indices_[i]);
              }
              SparseVecRef<f32, i16> query1(query.nnz_, indices.data(), query.data_);
              return index.SearchKnn(query1, topk, opt);
          });
  end_ts = std::chrono::high_resolution_clock::now();
  auto duration = end_ts - begin_ts;
  std::stringstream ss;
  i64 q_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
  i64 q_s = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
  double qps = query_mat.nrow_ / q_s;
  GTEST_LOG_(INFO) << "Total query counts: "<<query_mat.nrow_;
  GTEST_LOG_(INFO) << "Total query time: " << q_ms << " ms, QPS: " << qps;
  auto [topk, query_n, indices, scores] = DecodeGroundtruth("/home/qiyu.zd/sparse/data/base_1M.dev.gt");
  f32 recall = CheckGroundtruth(indices.get(), scores.get(), query_result, top_k);
  GTEST_LOG_(INFO) << "recall: " << recall; 


//   std::chrono::time_point<std::chrono::high_resolution_clock> begin_ts1, end_ts1;

//   Vector<Pair<Vector<u32>, Vector<f32>>> query_result1;

//   GTEST_LOG_(INFO) << "Start querying with "<<thread_n<<" thread, limit "<<top_k;
//   begin_ts1 = std::chrono::high_resolution_clock::now();

//   query_result1 = Search(thread_n,
//           query_mat,
//           top_k,
//           query_mat.nrow_,
//           opt,
//           [&](const SparseVecRef<f32, i32> &query, u32 topk, BmpSearchOptions &opt) -> Pair<Vector<u32>, Vector<f32>> {
//               Vector<i16> indices(query.nnz_);
//               for (i32 i = 0; i < query.nnz_; i++) {
//                   indices[i] = static_cast<i16>(query.indices_[i]);
//               }
//               SparseVecRef<f32, i16> query1(query.nnz_, indices.data(), query.data_);
//               return index.SearchKnn(query1, topk, opt);
//           });
//   end_ts1 = std::chrono::high_resolution_clock::now();
//   auto duration1 = end_ts1 - begin_ts1;

//   i64 q_ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
//   i64 q_s1 = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
//   double qps1 = query_mat.nrow_ / q_s1;
//   GTEST_LOG_(INFO) << "Total query counts: "<<query_mat.nrow_;
//   GTEST_LOG_(INFO) << "Total query time: " << q_ms << " ms, QPS: " << qps1;
//   auto [topk1, query_n1, indices1, scores1] = DecodeGroundtruth("/home/qiyu.zd/sparse/data/base_1M.dev.gt");
//   f32 recall1 = CheckGroundtruth(indices1.get(), scores1.get(), query_result1, top_k);
//   GTEST_LOG_(INFO) << "recall: " << recall1;

  opt.alpha_ = 0.9;
  opt.beta_ = 0.65;
  GTEST_LOG_(INFO) << "Start querying with "<<thread_n<<" thread, limit "<<top_k;
  begin_ts = std::chrono::high_resolution_clock::now();


  Vector<Pair<Vector<u32>, Vector<f32>>> query_result2;
  query_result2 = Search(thread_n,
          query_mat,
          top_k,
          query_mat.nrow_,
          opt,
          [&](const SparseVecRef<f32, i32> &query, u32 topk, BmpSearchOptions &opt) -> Pair<Vector<u32>, Vector<f32>> {
              Vector<i16> indices(query.nnz_);
              for (i32 i = 0; i < query.nnz_; i++) {
                  indices[i] = static_cast<i16>(query.indices_[i]);
              }
              SparseVecRef<f32, i16> query1(query.nnz_, indices.data(), query.data_);
              return index.SearchKnn(query1, topk, opt);
          });
  end_ts = std::chrono::high_resolution_clock::now();
  auto duration2 = end_ts - begin_ts;
  
  i64 q_ms2 = std::chrono::duration_cast<std::chrono::milliseconds>(duration2).count();
  i64 q_s2 = std::chrono::duration_cast<std::chrono::seconds>(duration2).count();
  double qps2 = query_mat.nrow_ / q_s2;
  GTEST_LOG_(INFO) << "Total query counts: "<<query_mat.nrow_;

  GTEST_LOG_(INFO) << "Total query time: " << q_ms2 << " ms, QPS: " << qps2;
  f32 recall2 = CheckGroundtruth(indices.get(), scores.get(), query_result2, top_k);
  GTEST_LOG_(INFO) << "recall: " << recall2; 

}
