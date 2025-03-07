/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
* Copyright 2020 Codeplay Software Limited
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_NVIDIA_CUDNN_BATCH_NORMALIZATION_HPP
#define GPU_NVIDIA_CUDNN_BATCH_NORMALIZATION_HPP

#include <cudnn.h>

#include "common/batch_normalization_pd.hpp"
#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/nvidia/cudnn_batch_normalization_executor.hpp"
#include "gpu/nvidia/cudnn_batch_normalization_impl.hpp"
#include "gpu/nvidia/engine.hpp"
#include "gpu/nvidia/stream.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_batch_normalization_common_t {
    template <typename pd_t>
    static status_t execute(
            const exec_ctx_t &ctx, impl::engine_t *engine, const pd_t *pd) {
        if (memory_desc_wrapper(pd->src_md()).has_zero_dim())
            return status::success;
        return pd->executor_->execute(ctx, engine, pd->bnorm_impl_);
    }

    template <typename pd_t>
    static void init_ws(const pd_t *pd, memory_desc_t &ws_md) {
        const auto wrap = memory_desc_wrapper(pd->src_md());
        const auto y_size
                = wrap.nelems() * types::data_type_size(data_type::f32);
        // Mean and variance are always f32.
        const size_t mean_invvar_size
                = 2 * pd->C() * types::data_type_size(data_type::f32);
        const dims_t ws_size
                = {(dim_t)(y_size * pd->fuse_norm_relu() + mean_invvar_size)};

        memory_desc_init_by_tag(
                ws_md, 1, ws_size, data_type::u8, format_tag::x);
    }
};

struct cudnn_batch_normalization_fwd_t : public gpu::primitive_t {
    using gpu::primitive_t::primitive_t;
    struct pd_t : public batch_normalization_fwd_pd_t {
        using batch_normalization_fwd_pd_t::batch_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_batch_normalization_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using namespace types;

            const auto norm_flags_supported
                    = normalization_flags::use_global_stats
                    | normalization_flags::fuse_norm_relu
                    | normalization_flags::use_scale
                    | normalization_flags::use_shift;
            if ((~norm_flags_supported & desc()->flags) != 0)
                return status::unimplemented;

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops;
            auto *sycl_engine_impl
                    = utils::downcast<const xpu::sycl::engine_impl_t *>(
                            engine->impl());

            bool ok = is_fwd()
                    && utils::one_of(src_md()->data_type, f16, f32, s8, bf16)
                    && src_md()->data_type == dst_md()->data_type
                    && check_scale_shift_data_type()
                    && attr()->has_default_values(attr_skip_mask)
                    && IMPLICATION(
                            utils::one_of(data_type::bf16, src_md()->data_type,
                                    dst_md()->data_type),
                            has_bf16_support(sycl_engine_impl->device()))
                    && IMPLICATION(!attr()->has_default_values(),
                            attr()->post_ops_.len() == 1 && with_relu_post_op())
                    && IMPLICATION(utils::one_of(src_md()->data_type, s8, f16),
                            !is_training() && stats_is_src())
                    && set_default_formats_common()
                    && memory_desc_wrapper(src_md())
                            == memory_desc_wrapper(dst_md())
                    && src_md()->format_desc.blocking.inner_nblks == 0;
            if (!ok) return status::unimplemented;

            if (is_training()) {
                cudnn_batch_normalization_common_t::init_ws(this, ws_md_);
            }

            if (use_global_stats()) {
                bnorm_impl_.reset(
                        new cudnn_batch_normalization_fwd_stats_impl_t());
            } else {
                bnorm_impl_.reset(new cudnn_batch_normalization_fwd_impl_t());
            }

            executor_.reset(new bnorm_exec_fwd_t());

            return bnorm_impl_->init(this);
        }

        std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl_;
        std::shared_ptr<bnorm_exec_base_t> executor_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct cudnn_batch_normalization_bwd_t : public gpu::primitive_t {
    using gpu::primitive_t::primitive_t;

    struct pd_t : public batch_normalization_bwd_pd_t {
        using batch_normalization_bwd_pd_t::batch_normalization_bwd_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_batch_normalization_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using namespace types;

            const auto norm_flags_supported
                    = normalization_flags::fuse_norm_relu
                    | normalization_flags::use_scale
                    | normalization_flags::use_shift;
            if ((~norm_flags_supported & desc()->flags) != 0)
                return status::unimplemented;
            auto *sycl_engine_impl
                    = utils::downcast<const xpu::sycl::engine_impl_t *>(
                            engine->impl());

            bool ok = !is_fwd()
                    && (utils::everyone_is(f32, src_md()->data_type,
                                diff_src_md()->data_type,
                                diff_dst_md()->data_type)
                            || utils::everyone_is(bf16, src_md()->data_type,
                                    diff_src_md()->data_type,
                                    diff_dst_md()->data_type))
                    && IMPLICATION(
                            utils::one_of(data_type::bf16, src_md()->data_type,
                                    diff_src_md()->data_type,
                                    diff_dst_md()->data_type),
                            has_bf16_support(sycl_engine_impl->device()))
                    && check_scale_shift_data_type()
                    && attr()->has_default_values()
                    && set_default_formats_common()
                    && memory_desc_wrapper(diff_src_md())
                            == memory_desc_wrapper(diff_dst_md())
                    && src_md()->format_desc.blocking.inner_nblks == 0
                    && diff_src_md()->format_desc.blocking.inner_nblks == 0;
            if (!ok) return status::unimplemented;

            cudnn_batch_normalization_common_t::init_ws(this, ws_md_);
            if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;

            if (fuse_norm_relu()) {
                bnorm_impl_.reset(
                        new cudnn_batch_normalization_bwd_relu_impl_t());
            } else {
                bnorm_impl_.reset(new cudnn_batch_normalization_bwd_impl_t());
            }

            executor_.reset(new bnorm_exec_bwd_t());

            return bnorm_impl_->init(this);
        }

        std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl_;
        std::shared_ptr<bnorm_exec_base_t> executor_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
