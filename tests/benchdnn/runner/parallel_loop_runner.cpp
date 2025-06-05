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

#include "parallel_loop_runner.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "work_queue.h"
#include "async_value_ref.h"
#include "chain.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

// namespace xla::cpu {

// Returns non-reference-counted async value ref in constructed state.
//
// Returned async value is a per-process singleton stored in a storage with a
// static duration, and can be safely compared using pointer equality.
static AsyncValueRef<Chain> OkDoneEventSingleton() {
  static AsyncValueOwningRef<Chain>* singleton = [] {
    auto* storage = new internal::AsyncValueStorage<Chain>();
    return new AsyncValueOwningRef<Chain>(
        MakeAvailableAsyncValueRef<Chain>(*storage));
  }();
  return singleton->AsRef();
}

ParallelLoopRunner::ParallelLoopRunner(const Eigen::ThreadPoolDevice* device)
    : done_event_(OkDoneEventSingleton()), device_(device) {}

AsyncValueRef<Chain> ParallelLoopRunner::ResetDoneEvent() {
  auto done_event = std::move(done_event_);
  done_event_ = OkDoneEventSingleton();
  return done_event;
}

size_t ParallelLoopRunner::num_threads() const {
  return device_.load()->numThreadsInPool();
}

bool ParallelLoopRunner::is_in_runner() const {
  return device_.load()->currentThreadId() > -1;
}

AsyncValueRef<Chain> ParallelLoopRunner::TakeDoneEvent(
    ParallelLoopRunner&& runner) {
  return std::move(runner.done_event_);
}

template <typename Task>
ABSL_ATTRIBUTE_ALWAYS_INLINE void ParallelLoopRunner::ScheduleOne(Task&& task) {
  auto event = MakeConstructedAsyncValueRef<Chain>();
  done_event_.AndThen([event, task = std::forward<Task>(task)] {
    task();
    event.SetStateConcrete();
  });
  done_event_ = std::move(event);
}

template <typename ParallelTask>
ABSL_ATTRIBUTE_ALWAYS_INLINE void ParallelLoopRunner::ScheduleAll(
    size_t num_tasks, ParallelTask&& parallel_task) {
  // DCHECK_GT(num_tasks, 1) << "Expected at least two task";

  // Use at most `num_threads()` workers as we can't run more parallel workers
  // than the number of threads in the thread pool.
  size_t num_workers = std::min(std::min(num_tasks, num_threads()),
                                size_t{std::numeric_limits<uint16_t>::max()});

  auto parallelize =
      [this, num_workers, num_tasks,
       parallel_task = std::forward<ParallelTask>(parallel_task)](Chain) {
        return Worker::Parallelize(device_.load()->getPool(), num_workers,
                                   num_tasks, std::move(parallel_task));
      };

  done_event_ = done_event_.FlatMap(parallelize);
}

// Parallelize `task` over dimensions `dims` using `ParallelTask`.
//
// (1) If done event is already available, execute the task immediately in the
//     caller thread. In this case we don't need to overwrite the done event,
//     because the existing one will correctly represent the state of the
//     parallel loop runner (all scheduled loops are ready).
//
// (2) If done event is not available, we have to overwrite it with a new one
//     that will be set to concrete state after the task is executed.
//
// We wrap all tasks into structs conforming to the `ParallelTest` API, so that
// in profiles we can see human-readable names of the tasks instead of lambdas.
template <typename ParallelTask, typename... Dims, typename Task>
ABSL_ATTRIBUTE_ALWAYS_INLINE void ParallelLoopRunner::Parallelize(Dims... dims,
                                                                  Task&& task) {
  // DCHECK(done_event_) << "Parallel loop runner is in moved-from state";

  size_t num_tasks = NumTasks(dims...);
  // DCHECK_GT(num_tasks, 0) << "Expected at least one task";

  // Fast path for the degenerate parallel loop with a single task.
  if (ABSL_PREDICT_TRUE(num_tasks == 1)) {
    // Converts the dimension into the first task index.
    auto to_first_task_index = [](auto dim) {
      if constexpr (std::is_same_v<decltype(dim), RangeDim>) {
        return RangeIndex{0};
      // } else {
      //   return TileIndex{0, dim.range};
      }
    };

    // Execute task in the caller thread if done event is already available.
    if (ABSL_PREDICT_TRUE(done_event_.IsConcrete())) {
      task(to_first_task_index(dims)...);
      return;
    }

    // Schedule task when done event becomes available.
    ScheduleOne([task = std::forward<Task>(task),
                 idxs = std::make_tuple(to_first_task_index(dims)...)] {
      std::apply([&task](auto... idxs) { task(idxs...); }, idxs);
    });
    return;
  }

  ScheduleAll(num_tasks, ParallelTask{dims..., std::forward<Task>(task)});
}

// void ParallelLoopRunner::Parallelize(RangeDim i, Task1D task) {
//   Parallelize<ParallelTask1D, RangeDim>(i, std::move(task));
// }

// }  // namespace xla::cpu
