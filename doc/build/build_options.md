Use Build Options {#dev_guide_build_options}
============================================

oneDNN supports the following build-time options.

| CMake Option                    | Supported values (defaults in bold)                 | Description                                                                                     |
|:--------------------------------|:----------------------------------------------------|:------------------------------------------------------------------------------------------------|
| ONEDNN_LIBRARY_TYPE             | **SHARED**, STATIC                                  | Defines the resulting library type                                                              |
| ONEDNN_CPU_RUNTIME              | NONE, **OMP**, TBB, SEQ, THREADPOOL, SYCL           | Defines the threading runtime for CPU engines                                                   |
| ONEDNN_GPU_RUNTIME              | **NONE**, OCL, SYCL                                 | Defines the offload runtime for GPU engines                                                     |
| ONEDNN_BUILD_DOC                | **ON**, OFF                                         | Controls building the documentation                                                             |
| ONEDNN_BUILD_EXAMPLES           | **ON**, OFF                                         | Controls building the examples                                                                  |
| ONEDNN_BUILD_TESTS              | **ON**, OFF                                         | Controls building the tests                                                                     |
| ONEDNN_BUILD_GRAPH              | **ON**, OFF                                         | Controls building graph component                                                               |
| ONEDNN_ENABLE_GRAPH_DUMP        | ON, **OFF**                                         | Controls dumping graph artifacts                                                                |
| ONEDNN_ARCH_OPT_FLAGS           | *compiler flags*                                    | Specifies compiler optimization flags (see warning note below)                                  |
| ONEDNN_ENABLE_CONCURRENT_EXEC   | ON, **OFF**                                         | Disables sharing a common scratchpad between primitives in #dnnl::scratchpad_mode::library mode |
| ONEDNN_ENABLE_JIT_PROFILING     | **ON**, OFF                                         | Enables [integration with performance profilers](@ref dev_guide_profilers)                      |
| ONEDNN_ENABLE_ITT_TASKS         | **ON**, OFF                                         | Enables [integration with performance profilers](@ref dev_guide_profilers)                      |
| ONEDNN_ENABLE_PRIMITIVE_CACHE   | **ON**, OFF                                         | Enables [primitive cache](@ref dev_guide_primitive_cache)                                       |
| ONEDNN_ENABLE_MAX_CPU_ISA       | **ON**, OFF                                         | Enables [CPU dispatcher controls](@ref dev_guide_cpu_dispatcher_control)                        |
| ONEDNN_ENABLE_CPU_ISA_HINTS     | **ON**, OFF                                         | Enables [CPU ISA hints](@ref dev_guide_cpu_isa_hints)                                           |
| ONEDNN_ENABLE_WORKLOAD          | **TRAINING**, INFERENCE                             | Specifies a set of functionality to be available based on workload                              |
| ONEDNN_ENABLE_PRIMITIVE         | **ALL**, PRIMITIVE_NAME                             | Specifies a set of functionality to be available based on primitives                            |
| ONEDNN_ENABLE_PRIMITIVE_CPU_ISA | **ALL**, CPU_ISA_NAME                               | Specifies a set of functionality to be available for CPU backend based on CPU ISA               |
| ONEDNN_ENABLE_PRIMITIVE_GPU_ISA | **ALL**, GPU_ISA_NAME                               | Specifies a set of functionality to be available for GPU backend based on GPU ISA               |
| ONEDNN_ENABLE_GEMM_KERNELS_ISA  | **ALL**, NONE, ISA_NAME                             | Specifies a set of functionality to be available for GeMM kernels for CPU backend based on ISA  |
| ONEDNN_EXPERIMENTAL             | ON, **OFF**                                         | Enables [experimental features](@ref dev_guide_experimental)                                    |
| ONEDNN_VERBOSE                  | **ON**, OFF                                         | Enables [verbose mode](@ref dev_guide_verbose)                                                  |
| ONEDNN_DEV_MODE                 | ON, **OFF**                                         | Enables internal tracing and `debuginfo` logging in verbose output (for oneDNN developers)      |
| ONEDNN_AARCH64_USE_ACL          | ON, **OFF**                                         | Enables integration with Arm Compute Library for AArch64 builds                                 |
| ONEDNN_BLAS_VENDOR              | **NONE**, ARMPL, ACCELERATE                         | Defines an external BLAS library to link to for GEMM-like operations                            |
| ONEDNN_GPU_VENDOR               | NONE, **INTEL**, NVIDIA, AMD                        | When DNNL_GPU_RUNTIME is not NONE defines GPU vendor for GPU engines otherwise its value is NONE|
| ONEDNN_DPCPP_HOST_COMPILER      | **DEFAULT**, *GNU or Clang C++ compiler executable* | Specifies host compiler executable for SYCL runtime                                             |
| ONEDNN_LIBRARY_NAME             | **dnnl**, *library name*                            | Specifies name of the library                                                                   |
| ONEDNN_TEST_SET                 | SMOKE, **CI**, NIGHTLY, MODIFIER_NAME               | Specifies the testing coverage enabled through the generated testing targets                    |

All building options listed support their counterparts with `DNNL` prefix
instead of `ONEDNN`. `DNNL` options would take precedence over `ONEDNN`
versions, if both versions are specified.

`ONEDNN_BUILD_DOC`, `ONEDNN_BUILD_EXAMPLES` and `ONEDNN_BUILD_TESTS` are disabled
by default when oneDNN is built as a sub-project.

All other building options or values that can be found in CMake files are
intended for development/debug purposes and are subject to change without
notice. Please avoid using them.

## Common options

### Host compiler
When building oneDNN with oneAPI DPC++/C++ Compiler user can specify a custom
host compiler. The host compiler is a compiler that will be used by the main
compiler driver to perform host compilation step.

The host compiler can be specified with `ONEDNN_DPCPP_HOST_COMPILER` CMake
option. It should be specified either by name (in this case, the standard system
environment variables will be used to discover it) or an absolute path to the
compiler executable.

The default value of `ONEDNN_DPCPP_HOST_COMPILER` is `DEFAULT`, which is the
default host compiler used by the compiler specified with `CMAKE_CXX_COMPILER`.

The `DEFAULT` host compiler is the only supported option on Windows.
On Linux, user can specify a GNU C++ compiler as the host compiler.

@warning
oneAPI DPC++/C++ Compiler requires host compiler to be compatible. The minimum
allowed GNU C++ compiler version is 7.4.0. See [GCC* Compatibility and Interoperability](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/gcc-compatibility-and-interoperability.html)
section in oneAPI DPC++/C++ Compiler Developer Guide.

@warning
The minimum allowed Clang C++ compiler version is 8.0.0.

### Configuring functionality
Using `ONEDNN_ENABLE_WORKLOAD` and `ONEDNN_ENABLE_PRIMITIVE` it is possible to
limit functionality available in the final shared object or statically linked
application. This helps to reduce the amount of disk space occupied by an app.

#### ONEDNN_ENABLE_WORKLOAD
This option supports only two values: `TRAINING` (the default) and `INFERENCE`.
`INFERENCE` enables only forward propagation kind part of functionality,
removing all backward-related functionality, except those which are
dependencies for forward propagation kind part.

#### ONEDNN_ENABLE_PRIMITIVE
This option supports several values: `ALL` (the default) which enables all
primitives implementations or a set of `BATCH_NORMALIZATION`, `BINARY`,
`CONCAT`, `CONVOLUTION`, `DECONVOLUTION`, `ELTWISE`, `GROUP_NORMALIZATION`,
`INNER_PRODUCT`, `LAYER_NORMALIZATION`, `LRN`, `MATMUL`, `POOLING`, `PRELU`,
`REDUCTION`, `REORDER`, `RESAMPLING`, `RNN`, `SDPA`, `SHUFFLE`, `SOFTMAX`,
`SUM`. When a set is used, only those selected primitives implementations will
be available. Attempting to use other primitive implementations will end up
returning an unimplemented status when creating primitive descriptor. In order
to specify a set, a CMake-style string should be used, with semicolon
delimiters, as in this example:
```
-DONEDNN_ENABLE_PRIMITIVE=CONVOLUTION;MATMUL;REORDER
```

#### ONEDNN_ENABLE_PRIMITIVE_CPU_ISA
This option supports several values: `ALL` (the default) which enables all
ISA implementations or one of `SSE41`, `AVX2`, `AVX512`, and `AMX`. Values are
linearly ordered as `SSE41` < `AVX2` < `AVX512` < `AMX`. When specified,
selected ISA and all ISA that are "smaller" will be available. When specified,
[CPU dispatcher controls](@ref dev_guide_cpu_dispatcher_control) are also
affected in compliance with the option.

Note that `AVX2` denotes whole AVX2-based family ISAs, `AVX512` denotes whole
AVX512-based family ISAs, as well as `AMX` denotes any ISA containing AMX unit.

Example that enables SSE41 and AVX2 sets:
```
-DONEDNN_ENABLE_PRIMITIVE_CPU_ISA=AVX2
```

#### ONEDNN_ENABLE_PRIMITIVE_GPU_ISA
This option supports several values: `ALL` (the default) which enables all
ISA implementations or any set of `XELP`, `XEHP`, `XEHPG`, `XEHPC`, `XE2`,
and `XE3`. Selected ISA will enable correspondent parts in just-in-time
kernel generation based implementations. OpenCL based kernels and
implementations will always be available. Example that enables XeLP and XeHP
set:
```
-DONEDNN_ENABLE_PRIMITIVE_GPU_ISA=XELP;XEHP
```

#### ONEDNN_ENABLE_GEMM_KERNELS_ISA
This option supports several values: `ALL` (the default) which enables all
ISA kernels from x64/gemm folder, `NONE` which disables all kernels and removes
correspondent interfaces, or one of `SSE41`, `AVX2`, and `AVX512`. Values are
linearly ordered as `SSE41` < `AVX2` < `AVX512`. When specified, selected ISA
and all ISA that are "smaller" will be available. Example that leaves SSE41 and
AVX2 sets, but removes AVX512 and AMX kernels:
```
-DONEDNN_ENABLE_GEMM_KERNELS_ISA=AVX2
```

### Configuring testing

#### ONEDNN_TEST_SET
This option specifies testing coverage enabled through testing targets generated
by the build system. The variable consists of two parts: the set value which
defines the number of test cases, and the modifiers for testing commands. The
final string must contain a single value for a set and as many compatible values
for modifiers.

The set value is defined by one of: `SMOKE`, `CI`, or `NIGHTLY`.
The modifier values (referred as `MODIFIER_NAME`) are one of: `NO_CORR`,
`ADD_BITWISE`.
The input is expected in the CMake list style - a semicolon separated string -
e.g., `ONEDNN_TEST_SET=CI;NO_CORR`.

When `SMOKE` value is specified, it enables a short set of test cases which
verifies that basic library functionality works as expected.
When `CI` value is specified, it enables a regular set of test cases which
verifies that all library supported functionality works as expected.
When `NIGHTLY` value is specified, it enables the largest set of test cases
which verifies that all library supported functionality and all kernel
optimizations work as expected.

When `NO_CORR` modifier value is specified, it removes correctness validation,
which is set by default, from benchdnn testing targets. It helps to save time
when correctness validation is not necessary.
When `ADD_BITWISE` modifier value is specified, the build system will add an
additional set of tests with a bitwise validation mode for benchdnn. The
correctness set remains unmodified.

## CPU Options
Intel Architecture Processors and compatible devices are supported by
oneDNN CPU engine. The CPU engine is built by default but can be disabled
at build time by setting `ONEDNN_CPU_RUNTIME` to `NONE`. In this case,
GPU engine must be enabled.

### Targeting Specific Architecture
oneDNN uses JIT code generation to implement most of its functionality
and will choose the best code based on detected processor features. However,
some oneDNN functionality will still benefit from targeting a specific
processor architecture at build time. You can use `ONEDNN_ARCH_OPT_FLAGS` CMake
option for this.

For Intel(R) C++ Compilers, the default option is `-xSSE4.1`, which instructs
the compiler to generate the code for the processors that support SSE4.1
instructions. This option would not allow you to run the library on
older processor architectures.

For GNU\* Compilers and Clang, the default option is `-msse4.1`.

@warning
While use of `ONEDNN_ARCH_OPT_FLAGS` option gives better performance, the
resulting library can be run only on systems that have instruction set
compatible with the target instruction set. Therefore, `ARCH_OPT_FLAGS`
should be set to an empty string (`""`) if the resulting library needs to be
portable.

### Runtimes
CPU engine can use OpenMP, Threading Building Blocks (TBB) or sequential
threading runtimes. OpenMP threading is the default build mode. This behavior
is controlled by the `ONEDNN_CPU_RUNTIME` CMake option.

#### OpenMP
oneDNN uses OpenMP runtime library provided by the compiler.

When building oneDNN with oneAPI DPC++/C++ Compiler the library will link
to Intel OpenMP runtime. This behavior can be changed by changing the host
compiler with `ONEDNN_DPCPP_HOST_COMPILER` option.

@warning
Because different OpenMP runtimes may not be binary-compatible, it's important
to ensure that only one OpenMP runtime is used throughout the application.
Having more than one OpenMP runtime linked to an executable may lead to
undefined behavior including incorrect results or crashes. However as long as
both the library and the application use the same or compatible compilers there
would be no conflicts.

#### Threading Building Blocks (TBB)
To build oneDNN with TBB support, set `ONEDNN_CPU_RUNTIME` to `TBB`:

~~~sh
$ cmake -DONEDNN_CPU_RUNTIME=TBB ..
~~~

Optionally, set the `TBBROOT` environmental variable to point to the TBB
installation path or pass the path directly to CMake:

~~~sh
$ cmake -DONEDNN_CPU_RUNTIME=TBB -DTBBROOT=/opt/intel/path/tbb ..
~~~

oneDNN has functional limitations if built with TBB:
* Winograd convolution algorithm is not supported for fp32 backward
  by data and backward by weights propagation.

#### Threadpool
To build oneDNN with support for threadpool threading, set `ONEDNN_CPU_RUNTIME`
to `THREADPOOL`

~~~sh
$ cmake -DONEDNN_CPU_RUNTIME=THREADPOOL ..
~~~

The `_ONEDNN_TEST_THREADPOOL_IMPL` CMake variable controls which of the three
threadpool implementations would be used for testing: `STANDALONE`, `TBB`, or
`EIGEN`. The latter two require also passing `TBBROOT` or `Eigen3_DIR` paths
to CMake. For example:

~~~sh
$ cmake -DONEDNN_CPU_RUNTIME=THREADPOOL -D_ONEDNN_TEST_THREADPOOL_IMPL=EIGEN -DEigen3_DIR=/path/to/eigen/share/eigen3/cmake ..
~~~

Threadpool threading support is experimental and has the same limitations as
TBB plus more:
* As threadpools are attached to streams which are only passed during
  primitive execution, work decomposition is performed statically at the
  primitive creation time. At the primitive execution time, the threadpool is
  responsible for balancing the static decomposition from the previous item
  across available worker threads.

### AArch64 Options

oneDNN includes experimental support for Arm 64-bit Architecture (AArch64).
By default, AArch64 builds will use the reference implementations throughout.
The following options enable the use of AArch64 optimised implementations
for a limited number of operations, provided by AArch64 libraries.

| AArch64 build configuration          | CMake Option              | Environment variables                    | Dependencies                                                                                                                 |
|:-------------------------------------|:--------------------------|:-----------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------|
| Arm Compute Library based primitives | ONEDNN_AARCH64_USE_ACL=ON | ACL_ROOT_DIR=*</path/to/ComputeLibrary>* | [Arm Compute Library](https://github.com/ARM-software/ComputeLibrary)                                                        |
| Vendor BLAS library support          | ONEDNN_BLAS_VENDOR=ARMPL  | None                                     | [Arm Performance Libraries](https://developer.arm.com/tools-and-software/server-and-hpc/downloads/arm-performance-libraries) |

#### Arm Compute Library
Arm Compute Library is an open-source library for machine learning applications.
The development repository is available from
[mlplatform.org](https://review.mlplatform.org/#/admin/projects/ml/ComputeLibrary),
and releases are also available on [GitHub](https://github.com/ARM-software/ComputeLibrary).
The `ONEDNN_AARCH64_USE_ACL` CMake option is used to enable Compute Library integration:

~~~sh
$ cmake -DONEDNN_AARCH64_USE_ACL=ON ..
~~~

This assumes that the environment variable `ACL_ROOT_DIR` is
set to the location of Arm Compute Library, which must be downloaded and built
independently of oneDNN.

@warning
For a debug build of oneDNN it is advisable to specify a Compute Library build
which has also been built with debug enabled.

@warning
oneDNN only supports builds with Compute Library v23.11 or later.

#### Vendor BLAS libraries
oneDNN can use a standard BLAS library for GEMM operations.
The `ONEDNN_BLAS_VENDOR` build option controls BLAS library selection, and
defaults to `NONE`. For AArch64 builds with GCC, use the
[Arm Performance Libraries](https://developer.arm.com/tools-and-software/server-and-hpc/downloads/arm-performance-libraries):

~~~sh
$ cmake -DONEDNN_BLAS_VENDOR=ARMPL ..
~~~

Additional options available for development/debug purposes. These options are
subject to change without notice, see
[`cmake/options.cmake`](https://github.com/uxlfoundation/oneDNN/blob/main/cmake/options.cmake)
for details.

## GPU Options
Intel Processor Graphics is supported by oneDNN GPU engine. GPU engine
is disabled in the default build configuration.

### Runtimes
To enable GPU support you need to specify the GPU runtime by setting
`ONEDNN_GPU_RUNTIME` CMake option. The default value is `"NONE"` which
corresponds to no GPU support in the library.

#### OpenCL\*
OpenCL runtime requires Intel(R) SDK for OpenCL\* applications. You can
explicitly specify the path to the SDK using `-DOPENCLROOT` CMake option.

~~~sh
$ cmake -DONEDNN_GPU_RUNTIME=OCL -DOPENCLROOT=/path/to/opencl/sdk ..
~~~

@anchor component_limitation
## Graph component limitations

The graph component can be enabled via the build option `ONEDNN_BUILD_GRAPH`.
But the build option does not work with some values of other build options.
Specifying the options and values simultaneously in one build will lead to a
CMake error.

| CMake Option            | Unsupported Values |
|:------------------------|:-------------------|
| ONEDNN_ENABLE_PRIMITIVE | PRIMITIVE_NAME     |
