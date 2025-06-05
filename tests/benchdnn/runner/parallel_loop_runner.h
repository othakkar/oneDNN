/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_BACKENDS_CPU_RUNTIME_PARALLEL_LOOP_RUNNER_H_
#define XLA_BACKENDS_CPU_RUNTIME_PARALLEL_LOOP_RUNNER_H_

#include <array>
#include <atomic>
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "async_value_ref.h"
#include "chain.h"
#include "math_util.h"

namespace Eigen {
struct ThreadPoolDevice;
}  // namespace Eigen

// namespace xla::cpu {

// Parallel loop runner uses underlying Eigen ThreadPoolDevice to execute
// parallel loops providing implicit synchronization: the next parallel loop
// starts execution only after all tasks from the previous loop are completed.
//
// Scheduled parallel loops execute asynchronously without blocking the caller
// thread. It is the user's responsibility to ensure that all values captured by
// the task are valid until the task is completed.
//
// Parallel loop runner is an implementation of the `pthreadpool` API adaptor
// for XLA:CPU runtime.
//
// ParallelLoopRunner uses "persistent workers" to execute parallel loops.
// Workers get scheduled into the underlying thread pool and when they start
// executing they pop tasks from the shared work queue. With this approach we
// avoid scheduling closures into the thread pool for each parallel task,
// because fixed thread pool overheads are high and XNNPACK operations tend to
// launch many parallel loops with larget number of very small tasks.
//
// Parallel loop runner can be configured by the `worker_timeslice` parameter,
// that defines the approximate amount of compute (in terms of wall time) that
// each persistent worker will handle. We rely on this parameter to avoid
// scheduling too many workers into the thread pool, because for tiny tasks the
// overheads can be prohibitively expensive.
//
// WARNING: ParallelLoopRunner is not thread-safe, and must be externally
// synchronized by the user.
class ParallelLoopRunner {
 public:
  explicit ParallelLoopRunner(const Eigen::ThreadPoolDevice* device);

  // Takes ownership of the runner and returns a done event. After the done
  // event is transferred to the caller, it is illegal to schedule more parallel
  // loops on the moved-from runner.
  static AsyncValueRef<Chain> TakeDoneEvent(
      ParallelLoopRunner&& runner);

  //===--------------------------------------------------------------------===//
  // Parallel dimensions and task coordinates APIs.
  //===--------------------------------------------------------------------===//

  // Parallel dimension iterated in [0, range) range in parallel.
  struct RangeDim {
    size_t range;
  };

  // Parallel task index along the range dimension.
  struct RangeIndex {
    size_t offset;
  };

  // Mapping from parallel loop dimension to the parallel task index. Defined
  // as template specializations below.
  template <typename Dim>
  struct TaskIndex;

  static size_t DimSize(RangeDim dim) { return dim.range; }

  // Returns the number of tasks to be launched for the given dimensions.
  template <typename... Dims>
  static size_t NumTasks(Dims... dims);

  // Delinearizes linear `task_index` into the parallel task coordinates.
  template <typename... Dims>
  static std::tuple<typename TaskIndex<Dims>::Index...> Delinearize(
      size_t task_index, Dims... dims);

  //===--------------------------------------------------------------------===//
  // Parallel loop APIs.
  //===--------------------------------------------------------------------===//

  using Task1D = std::function<void(RangeIndex i)>;

  // IMPORTANT: For `dynamic` versions of the parallel loops, the runner is free
  // to adjust `count` for tiled dimensions to minimize the number of launched
  // tasks. Today we don't take advantage of this feature, and always launch the
  // same number of tasks as in regular parallel loops.

  // Launches `task` in parallel for each element of the `i` dimension.
  void Parallelize(RangeDim i, Task1D task);

  // Resets the parallel loop runner `done_event` and returns the previous one
  // to the caller.
  AsyncValueRef<Chain> ResetDoneEvent();

  AsyncValueRef<Chain> done_event() const { return done_event_; }

  const Eigen::ThreadPoolDevice* device() const { return device_; }
  void set_device(const Eigen::ThreadPoolDevice* device) { device_ = device; }

  // Returns the number of threads in the underlying thread pool.
  size_t num_threads() const;

  // Returns true if the current thread belongs to the underlying thread pool.
  bool is_in_runner() const;

 private:
  // Forward declarations of the parallel tasks.
  struct ParallelTask1D;

  // Schedules `task` as the AndThen callback of the `done_event_`. Updates
  // `done_event_` to the new completion event.
  template <typename Task>
  void ScheduleOne(Task&& task);

  // Schedules `num_tasks` invocation of the `parallel_task` into the Eigen
  // thread pool when the `done_event_` becomes available. Updates `done_event_`
  // to the new completion event.
  template <typename ParallelTask>
  void ScheduleAll(size_t num_tasks, ParallelTask&& parallel_task);

  // Internal implementation of the parallel loop APIs.
  template <typename ParallelTask, typename... Dims, typename Task>
  void Parallelize(Dims... dims, Task&& task);

  // Async value that signals completion of the last scheduled parallel loop.
  AsyncValueRef<Chain> done_event_;

  // We keep a pointer to the Eigen thread pool device as an atomic variable
  // because we might update it between concurrent runs of XNNPACK operations
  // and non-atomic access to the `device_` pointer might lead to a data race.
  //
  // In practice PjRt CPU client owns the intra-op thread pool and passes it to
  // XLA via Thunk::ExecuteParams, and PjRt client might have multiple thread
  // pools for different NUMA nodes, and we have to be able to switch between
  // them from run to run.
  std::atomic<const Eigen::ThreadPoolDevice*> device_;
};

// An explicit specialization shall be declared in the namespace of which the
// template is a member, or, for member templates, in the namespace of which the
// enclosing class or enclosing class template is a member.

template <>
struct ParallelLoopRunner::TaskIndex<ParallelLoopRunner::RangeDim> {
  using Index = RangeIndex;
};

//===----------------------------------------------------------------------===//
// Parallel dimensions and task coordinates APIs.
//===----------------------------------------------------------------------===//

namespace internal {

template <typename Dim>
auto TaskStrides(Dim dim) {
  return std::array<size_t, 1>{1};
}

template <typename Dim, typename... Dims>
auto TaskStrides(Dim dim, Dims... dims) {
  std::array<size_t, 1 + sizeof...(Dims)> strides = {
      ParallelLoopRunner::NumTasks(dims...)};
  absl::c_copy(TaskStrides(dims...), &strides[1]);
  return strides;
}

template <size_t n>
auto TaskCoordinate(size_t task_index, std::array<size_t, n> strides) {
  std::array<size_t, n> coordinate;
  for (size_t d = 0; d < n; ++d) {
    coordinate[d] = task_index / strides[d];
    task_index %= strides[d];
  }
  return coordinate;
}

}  // namespace internal

template <typename... Dims>
size_t ParallelLoopRunner::NumTasks(Dims... dims) {
  return (DimSize(dims) * ...);
}

template <typename... Dims>
std::tuple<typename ParallelLoopRunner::TaskIndex<Dims>::Index...>
ParallelLoopRunner::Delinearize(size_t task_index, Dims... dims) {
  // Convert linear task index into the multidimensional parallel task index.
  auto strides = internal::TaskStrides(dims...);
  auto coord = internal::TaskCoordinate(task_index, strides);

  size_t d = 0;
  auto to_task_index = [&](auto dim) {
    size_t dim_index = coord[d++];
    DCHECK_LE(dim_index, DimSize(dim)) << "Dimension index is out of bounds";

    if constexpr (std::is_same_v<decltype(dim), RangeDim>) {
      return RangeIndex{dim_index};
    } else {
      static_assert(sizeof(decltype(dim)) == 0, "Unsupported dimension type");
    }
  };

  return std::make_tuple(to_task_index(dims)...);
}

constexpr bool operator==(ParallelLoopRunner::RangeDim a,
                          ParallelLoopRunner::RangeDim b) {
  return a.range == b.range;
}

constexpr bool operator==(ParallelLoopRunner::RangeIndex a,
                          ParallelLoopRunner::RangeIndex b) {
  return a.offset == b.offset;
}

template <typename Sink>
void AbslStringify(Sink& sink, ParallelLoopRunner::RangeDim dim) {
  absl::Format(&sink, "RangeDim{range=%zu}", dim.range);
}

template <typename Sink>
void AbslStringify(Sink& sink, ParallelLoopRunner::RangeIndex index) {
  absl::Format(&sink, "RangeIndex{offset=%zu}", index.offset);
}

// }  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_PARALLEL_LOOP_RUNNER_H_
