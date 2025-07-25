/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#include "cpu/x64/brgemm/brgemm_utils.hpp"
#include "cpu/x64/brgemm/jit_brdgmm_kernel.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;

enum {
    decomposition_2x2 = 101,
    decomposition_3x1_3,
    decomposition_3x1_2,
    undefined,
};

impl::data_type_t get_accum_datatype(brgemm_desc_t *brg) {
    // this assert should check if 'init_kernel_datatype()' was previously
    // called.
    assert(brg->is_int8 || brg->is_bf16 || brg->is_f32 || brg->is_f16
            || brg->is_fp8);
    return brg->is_int8 ? data_type::s32 : data_type::f32;
}

status_t init_kernel_datatype(
        brgemm_desc_t *brg, impl::data_type_t dt_a, impl::data_type_t dt_b) {
    if (utils::one_of(data_type::undef, dt_a, dt_b)) {
        assert(!"Unsupported data type");
        return status::unimplemented;
    }

    brg->is_int8 = utils::one_of(dt_a, data_type::u8, data_type::s8)
            && utils::one_of(dt_b, data_type::u8, data_type::s8);
    brg->is_bf16 = (dt_a == data_type::bf16) && (dt_b == data_type::bf16);
    // Note: f32:bf16 is treated as f32 case while f32:f16 has already been
    // treated as f16. Probably, need a common ground here.
    brg->is_f32 = (dt_a == data_type::f32)
            && utils::one_of(
                    dt_b, data_type::f32, data_type::bf16, data_type::f16);
    brg->is_f16 = utils::one_of(data_type::f16, dt_a, dt_b) && !brg->is_f32;
    brg->is_fp8 = one_of(dt_a, data_type::f8_e5m2, data_type::f8_e4m3)
            && one_of(dt_b, data_type::f8_e5m2, data_type::f8_e4m3);
    if (utils::everyone_is(false, brg->is_int8, brg->is_bf16, brg->is_f32,
                brg->is_f16, brg->is_fp8)) {
        assert(!"Unsupported data type");
        return status::unimplemented;
    }
    return status::success;
}

void init_common_conf(brgemm_desc_t *brg, brgemm_batch_kind_t type, float alpha,
        float beta, const brgemm_strides_t *strides) {
    brg->beta = beta;
    brg->alpha = alpha;
    brg->type = type;
    brg->with_bias = false;
    brg->with_eltwise = false;
    brg->with_sum = false;
    brg->with_weights_scale_adjust = false;
    brg->sum_scale = 0;
    brg->sum_zp = 0;
    brg->with_scales = false;

    if (strides != nullptr) {
        brg->stride_a = strides->stride_a;
        brg->stride_b = strides->stride_b;
    } else {
        brg->stride_a = brg->stride_b = 0;
    }
}

namespace brgemm_utils {

bool can_dispatch_uker(const brgemm_desc_t *brg) {
    return brg->is_tmm
            && one_of(brg->type, brgemm_addr, brgemm_offs, brgemm_static_offs)
            && brg->brgattr.use_uker
            && everyone_is(false, brg->is_runtime_lda, brg->is_runtime_ldb,
                    brg->is_runtime_ldc, brg->is_runtime_ldd);
}

void maybe_try_bf32(brgemm_desc_t *brg) {
    const bool try_bf32 = brg->is_f32
            && one_of(brg->brgattr.fpmath_mode, fpmath_mode::bf16,
                    fpmath_mode::any)
            && one_of(brg->isa_user, isa_undef, avx512_core_amx)
            && mayiuse(avx512_core_amx);
    if (try_bf32) {
        const bool is_tmm = brg->is_tmm;
        brg->is_tmm = true;
        if (can_dispatch_uker(brg) /*Requires is_tmm to be true*/) {
            brg->is_bf32 = true;
        } else {
            brg->is_bf32 = false;
            //  Restore
            brg->is_tmm = is_tmm;
        }
    }
}

void set_isa_impl(brgemm_desc_t *brg) {
    auto is_isa_ok = [&](cpu_isa_t isa) {
        return mayiuse(isa) &&
                // maybe IMPLICATION(brg->isa_user != isa_undef,
                //  is_superset(brg->isa_user, isa)), but the API is not clear.
                one_of(brg->isa_user, isa_undef, isa);
    };

    if (brg->is_tf32) {
        brg->isa_impl = avx10_2_512_amx_2;
    } else if (brg->is_bf32) {
        brg->isa_impl = avx512_core_amx;
    } else if (brg->is_f32) {
        brg->isa_impl = utils::map(true, isa_undef,
                is_isa_ok(avx512_core) || is_isa_ok(avx512_core_amx) /*bf32*/,
                avx512_core, is_isa_ok(avx2), avx2,
                // Allow avx512_core_fp16 isa in case of a f16 primitive that
                // is implemented using pre-conversion of inputs to f32.
                // This is needed to support f16 binary post-ops.
                is_isa_ok(avx512_core_fp16), avx512_core_fp16, is_isa_ok(avx2),
                avx2);
    } else if (brg->is_bf16) {
        if (brg->dt_a == data_type::f32 && brg->dt_b == data_type::bf16) {
            // Distinguish f32:bf16 case upconversion for bf16 on AVX512_CORE
            // and AVX2.
            brg->isa_impl = utils::map(true, isa_undef,
                    is_isa_ok(avx512_core_amx), avx512_core_amx,
                    is_isa_ok(avx512_core_bf16), avx512_core_bf16,
                    is_isa_ok(avx512_core), avx512_core, is_isa_ok(avx2_vnni_2),
                    avx2_vnni_2, is_isa_ok(avx2), avx2);
        } else {
            brg->isa_impl = utils::map(true, isa_undef,
                    is_isa_ok(avx512_core_amx), avx512_core_amx,
                    is_isa_ok(avx512_core_bf16), avx512_core_bf16,
                    is_isa_ok(avx2_vnni_2), avx2_vnni_2);
        }
    } else if (brg->is_f16) {
        if (everyone_is(data_type::f16, brg->dt_a, brg->dt_b)) {
            brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(avx10_2_512),
                    avx10_2_512, is_isa_ok(avx512_core_amx_fp16),
                    avx512_core_amx_fp16, is_isa_ok(avx512_core_fp16),
                    avx512_core_fp16, is_isa_ok(avx2_vnni_2), avx2_vnni_2);
        } else if (brg->dt_a == data_type::f32 && brg->dt_b == data_type::f16) {
            // Distinguish f32:f16 case upconversion for f16 on AVX512_CORE and
            // AVX2.
            brg->isa_impl = utils::map(true, isa_undef,
                    is_isa_ok(avx512_core_fp16), avx512_core_fp16,
                    is_isa_ok(avx512_core), avx512_core, is_isa_ok(avx2), avx2);
        } else {
            brg->isa_impl = utils::map(true, isa_undef,
                    is_isa_ok(avx512_core_fp16), avx512_core_fp16);
        }
    } else if (brg->is_int8) {
        brg->isa_impl
                = utils::map(true, isa_undef, is_isa_ok(avx10_2_512_amx_2),
                        avx10_2_512_amx_2, is_isa_ok(avx512_core_amx_fp16),
                        avx512_core_amx_fp16, is_isa_ok(avx512_core_amx),
                        avx512_core_amx, is_isa_ok(avx10_2_512), avx10_2_512,
                        is_isa_ok(avx512_core_fp16), avx512_core_fp16,
                        is_isa_ok(avx512_core_vnni), avx512_core_vnni,
                        is_isa_ok(avx512_core), avx512_core,
                        is_isa_ok(avx2_vnni_2), avx2_vnni_2,
                        is_isa_ok(avx2_vnni), avx2_vnni, is_isa_ok(avx2), avx2);
    } else if (brg->is_fp8) {
        brg->isa_impl = utils::map(true, isa_undef,
                is_isa_ok(avx10_2_512_amx_2), avx10_2_512_amx_2,
                is_isa_ok(avx10_1_512_amx_fp16), avx10_1_512_amx_fp16,
                is_isa_ok(avx10_2_512), avx10_2_512);
    }
}

void set_brg_vmm(brgemm_desc_t *brg) {
    brg->is_tmm = brg->is_int8_tmm || brg->is_bf16_tmm || brg->is_f16_tmm
            || brg->is_bf32 || brg->is_fp8_tmm || brg->is_tf32;
    brg->is_zmm = !brg->is_tmm && mayiuse(avx512_core)
            && is_superset(brg->isa_impl, avx512_core);
    brg->is_ymm
            = !brg->is_zmm && mayiuse(avx2) && is_superset(brg->isa_impl, avx2);
}

int calculate_ldb_params(brgemm_desc_t *brg, const int try_ld_block2) {
    brg->ld_block2 = try_ld_block2;
    brg->ldb2 = brg->ldb / brg->ld_block2;
    brg->ldb2_tail = brg->ldb % brg->ld_block2;

    if (brg->ldb2 == 0) brg->ld_block2 = nstl::max(1, brg->ldb2_tail);
    brg->embd_bcst = brg->is_f32
            && (brg->ldb2_tail <= 1 && brg->ldb2 == 0)
            /*only avx512 or more can bcast*/
            && is_superset(brg->isa_impl, avx512_core);

    const int adj_ld_block2
            = (brg->ldb2 != 0) ? brg->ld_block2 : brg->ldb2_tail;
    return nstl::max(1, adj_ld_block2);
}

int calculate_max_bcast_block(brgemm_desc_t *brg, const int adj_ld_block2) {

    // TODO: Calculating the number of available registers should be re-factored
    // to use one code here and in brgemm kernel generator on
    // "max_effective_vregs" calculation
    int max_isa_regs = isa_num_vregs(brg->isa_impl);
    const int max_bcst_regs = brg->n_bcast_1_load ? 0 : 1;
    const int load_regs = brg->n_bcast_1_load ? 1 : adj_ld_block2;
    const bool req_zp_a_comp_pads
            = (brg->req_cal_comp_pads || brg->brgattr.max_top_vpad > 0
                      || brg->brgattr.max_bottom_vpad > 0)
            && brg->zp_type_a != brgemm_broadcast_t::none;

    // --------------  whole kernel --------------
    // To support the f16 vnni B matrix on non-AMX we need to use two Vmm
    // registers for permutation in brgemm kernel:
    // see f16_perm_even_vreg_ and f16_perm_odd_vreg_ in brgemm kernel
    const int b_vnni_regs = brg->is_f16_b_non_amx_vnni() ? 2 : 0;

    // non-VNNI INT8 dot product required 2 temp vectors:
    // see int8_ones_words() and int8_dot_product_temp() in brgemm kernel
    const int non_int8_vnni_regs
            = (brg->is_int8 && !brg->has_int8_vnni) ? 2 : 0;

    // non-AMX fp8 via conversion requires five registers
    // to convert fp8 to f16 vnni before dot product
    // see vmm_fp8_emu_aux* in brgemm kernel
    const int fp8_emu_regs = brg->is_fp8_via_convert_non_amx() ? 5 : 0;

    max_isa_regs -= b_vnni_regs + non_int8_vnni_regs + fp8_emu_regs;

    // --------------- microkernel ---------------
    // see vmm_inp_shift() in brgemm kernel
    const int compensation_regs = brg->req_s8s8_compensation
                    || brg->zp_type_a != brgemm_broadcast_t::none
            ? 1
            : 0;

    // see vmm_zp_a_shift(), vmm_one_bytes() in brgemm kernel
    const int zp_a_comp_pads_regs = req_zp_a_comp_pads ? 2 : 0;

    const int microkernel_regs = zp_a_comp_pads_regs + compensation_regs;

    const auto microkernel_max_reg_count
            = max_isa_regs - microkernel_regs - load_regs - max_bcst_regs;

    auto microkernel_max_bcast_block
            = microkernel_max_reg_count / (adj_ld_block2 + brg->n_bcast_1_load);

    // ----- post-ops and store accumulators -----
    const int beta_regs = !one_of(brg->beta, 1.f, 0.f);

    const int postops_regs = brg->attr()
            ? injector::aux_vec_count(
                    brg->attr()->post_ops_, brg->isa_impl, true)
            : 0;

    // Emulators: fp8 emulation are supported for amx only
    // In theory, vmm bf16_emu register indices overlap with other vmm
    // registers related to 'max_bcast_block'
    assert(IMPLICATION(
            brg->is_bf16_emu, is_superset(brg->isa_impl, avx512_core)));
    const int bf16_emu_regs = brg->is_bf16_emu ? 4 : 0;

    const auto store_regs = nstl::max(beta_regs,
            nstl::max(
                    postops_regs, nstl::max(compensation_regs, bf16_emu_regs)));

    const auto store_max_reg_count = max_isa_regs - store_regs;

    auto store_max_bcast_block = store_max_reg_count / adj_ld_block2;

    // ------------ final calculation ------------
    const auto max_bcast_block
            = nstl::min(microkernel_max_bcast_block, store_max_bcast_block);

    return max_bcast_block;
}

status_t brgemm_blocking_tmm(brgemm_desc_t *brg) {
    const auto L1 = platform::get_per_core_cache_size(1);

    // Blocking configuration for AMX
    const auto BD = brg->bcast_dim;
    const auto BD_R16 = rnd_up(BD, 16);
    const auto LD = brg->load_dim;
    const auto LD_R16 = rnd_up(LD, 16);

    const int max_width = 16, min_width = 1;
    brg->ld_block = 16;
    brg->ldb = LD / brg->ld_block;
    brg->ldb_tail = LD % brg->ld_block;

    auto find_bdb_bd_mask = [&](int bd_block, int &bdb, int &bdb_tail) {
        if (brg->brgattr.bd_mask_level != 2 || BD == 0) {
            bdb = div_up(BD, bd_block);
            bdb_tail = BD % bd_block;
            return;
        }

        bdb = 0;
        bdb_tail = 0;
        for (int i = 0; i < BD;) {
            if (brg->brgattr.bd_mask_level == 2
                    && brg->brgattr.bd_mask[i] == 0) {
                i++;
            } else {
                i += bd_block;
                if (i > BD) {
                    bdb_tail = BD - i + bd_block;
                    if (brg->brgattr.use_uker) bdb++;
                } else
                    bdb++;
            }
        }
    };

    auto find_bd_block_for_bd_mask = [&]() {
        if (brg->brgattr.bd_mask_level != 2 || BD == 0) return false;

        auto min_bdb = INT_MAX;
        const auto start_bd_block = nstl::min(max_width, BD);
        auto best_bd_block = start_bd_block;
        for (auto bd_block = start_bd_block; bd_block > 0; bd_block--) {
            int bdb = 0;
            int bdb_tail = 0;
            find_bdb_bd_mask(bd_block, bdb, bdb_tail);
            // bcast_dim should be divided by bd_block
            if (bdb < min_bdb && bdb_tail == 0) {
                min_bdb = bdb;
                best_bd_block = bd_block;
            }
        }
        brg->bd_block = best_bd_block;
        brg->bdb_tail = 0;
        brg->bdb = min_bdb;
        return true;
    };

    auto set_decomposition_by_ld = [&]() {
        if (brg->bd_block2 == 1 && brg->ldb > 0 && brg->ldb_tail == 0) {
            if (brg->ldb % 3 == 0)
                brg->ld_block2 = 3;
            else if (brg->ldb % 2 == 0)
                brg->ld_block2 = 2;
            else
                brg->ld_block2 = 1;
        } else {
            brg->ld_block2
                    = (brg->ldb > 0 && brg->ldb % 2 == 0 && brg->ldb_tail == 0
                              && brg->bd_block2 < 3)
                    ? 2
                    : 1;
        }
        brg->ldb2 = brg->ldb / brg->ld_block2;
        brg->ldb2_tail = brg->ldb % brg->ld_block2;

        // Re-adjust the bd_block2 if possible
        if (brg->ld_block2 == 1 && !brg->is_M_tail && brg->ldb_tail == 0) {
            brg->bd_block2 = (brg->bdb >= 3) ? 3 : (brg->bdb >= 2) ? 2 : 1;
            brg->bdb2 = brg->bdb / brg->bd_block2;
            brg->bdb2_tail = (brg->bd_block2 == 1) ? brg->bdb
                                                   : brg->bdb % brg->bd_block2;
        }
    };

    auto try_3x1_decomposition = [&](int width_step) {
        brg->is_M_tail = false;
        if (BD > (width_step - 1) * max_width && BD < width_step * max_width
                && brg->ldb_tail == 0) {
            if (!find_bd_block_for_bd_mask()) {
                brg->bd_block = max_width;
                brg->bdb = div_up(BD, brg->bd_block);
                brg->bdb_tail = BD % brg->bd_block;
                brg->is_M_tail = true;
            }
            brg->bd_block2 = width_step;
            brg->bdb2 = brg->bdb / brg->bd_block2;
            brg->bdb2_tail = brg->bdb % brg->bd_block2;
            set_decomposition_by_ld();
            return true;
        }
        return false;
    };

    auto try_2x2_decomposition = [&]() {
        if (!find_bd_block_for_bd_mask()) {
            for (int m_block = max_width; m_block >= min_width; m_block--) {
                if (BD % m_block == 0) {
                    brg->bd_block = m_block;
                    break;
                }
            }
            if (brg->bd_block == 1) {
                brg->bd_block = nstl::min(max_width, BD);
                brg->bdb_tail = BD % max_width;
                for (int i = max_width; i >= min_width; i--) {
                    const auto i_tail = BD % i;
                    if (i_tail > brg->bdb_tail || i_tail == 0) {
                        brg->bd_block = i;
                        brg->bdb_tail = i_tail;
                        if (i_tail == 0) break;
                    }
                }
            }
            brg->bdb = BD / brg->bd_block;
            brg->bdb_tail = BD % brg->bd_block;
        }

        brg->bd_block2 = (brg->bdb >= 2) ? 2 : 1;
        brg->bdb2 = brg->bdb / brg->bd_block2;
        brg->bdb2_tail
                = (brg->bd_block2 == 1) ? brg->bdb : brg->bdb % brg->bd_block2;

        brg->is_M_tail = false;

        set_decomposition_by_ld();

        return !(brg->ld_block2 == 1 || brg->bd_block2 == 1
                || brg->bd_block < 8);
    };

    auto recalc_blocking = [&](int new_bd_block, int new_ld_block,
                                   int new_bd_block2, int new_ld_block2) {
        if (new_bd_block != 0) {
            brg->bd_block = new_bd_block;
            find_bdb_bd_mask(brg->bd_block, brg->bdb, brg->bdb_tail);
            brg->is_M_tail = (brg->bdb_tail != 0);
        }

        if (new_ld_block != 0) {
            brg->ld_block = new_ld_block;
            brg->ldb = div_up(LD, brg->ld_block);
            brg->ldb_tail = LD % brg->ld_block;
        }

        if (new_bd_block2 != 0) {
            brg->bd_block2 = new_bd_block2;
            if (can_dispatch_uker(brg)) {
                brg->bdb2 = div_up(brg->bdb, brg->bd_block2);
                brg->bdb2_tail = 0;
            } else {
                if (brg->bdb_tail && brg->bd_block2 > 1) brg->bd_block2--;
                auto full_bd_blocks = brg->bdb - (brg->bdb_tail != 0 ? 1 : 0);
                brg->bdb2 = full_bd_blocks / brg->bd_block2;
                brg->bdb2_tail = full_bd_blocks % brg->bd_block2;
            }
        }

        if (new_ld_block2 != 0) {
            brg->ld_block2 = new_ld_block2;
            if (can_dispatch_uker(brg)) {
                brg->ldb2 = div_up(brg->ldb, brg->ld_block2);
                brg->ldb2_tail = 0;
            } else {
                if (brg->ldb_tail && brg->ld_block2 > 1) brg->ld_block2--;
                auto full_ld_blocks = brg->ldb - (brg->ldb_tail != 0 ? 1 : 0);
                brg->ldb2 = full_ld_blocks / brg->ld_block2;
                brg->ldb2_tail = full_ld_blocks % brg->ld_block2;
            }
        }
    };

    auto recalc_blocking_ext
            = [&](int new_bd_block, int new_ld_block, int new_bd_block2,
                      int new_ld_block2, bool load_nt_A, bool load_nt_B,
                      brgemm_kernel_innermost_loop_t innermost_loop) {
                  recalc_blocking(new_bd_block, new_ld_block, new_bd_block2,
                          new_ld_block2);
                  brg->load_nt_A = load_nt_A;
                  brg->load_nt_B = load_nt_B;
                  brg->innermost_loop = innermost_loop;
              };

    bool is_decomposition_defined = false;
    for (int i = decomposition_2x2; i != undefined; i++) {
        switch (i) {
            case decomposition_2x2:
                is_decomposition_defined = try_2x2_decomposition();
                break;
            case decomposition_3x1_3:
                is_decomposition_defined = try_3x1_decomposition(3);
                break;
            case decomposition_3x1_2:
                is_decomposition_defined = try_3x1_decomposition(2);
                break;
            default: assert(!"invalid value"); break;
        };
        if (is_decomposition_defined) break;
    }
    if (!is_decomposition_defined) try_2x2_decomposition();

    const bool try_load_nt_A
            = (brg->innermost_loop == brgemm_bd_loop_innermost);
    const bool try_load_nt_B
            = (brg->innermost_loop == brgemm_ld_loop_innermost);
    const bool try_load_nt
            = (static_cast<size_t>(brg->typesize_A)
                              * brg->brgattr.hint_expected_A_size
                      + static_cast<size_t>(brg->typesize_B)
                              * brg->brgattr.hint_expected_B_size
                      + static_cast<size_t>(brg->typesize_C)
                              * brg->brgattr.hint_expected_C_size)
            >= L1;
    brg->load_nt_A = try_load_nt_A && try_load_nt;
    brg->load_nt_B = try_load_nt_B && try_load_nt;

    recalc_blocking(
            brg->bd_block, brg->ld_block, brg->bd_block2, brg->ld_block2);

    if (can_dispatch_uker(brg)) {
        // Blocking heuristics for some shapes
        // TODO: Review these criteria
        const size_t eff_K
                = brg->reduce_dim * brg->typesize_A * brg->brgattr.K_koef;
        const auto low_K = (L1 - 4 * 1024) / (6 * 16);

        // TODO: if rdb_tail != 0 then we should limit
        // blocking because we need extra tiles for A and B to load rdb_tail
        // if bd_mask_level != 0 it means it aligned to 16

        const bool bdb_block_tail = !(brg->bd_block > 12
                && (BD % brg->bd_block == 0
                        && brg->brgattr.bd_mask_level == 0));
        const bool ldb_tail_16 = (LD % 16 != 0);
        if (everyone_is(false, bdb_block_tail, ldb_tail_16)) {
            // try to use 1x(4|5) or (4|5)x1 decomposition for specific
            // range of K
            const auto upper_K5 = (L1 - 5 * 1024) / (5 * 16);
            const auto upper_K4 = (L1 - 4 * 1024) / (4 * 16);
            const bool K5_fit_L1 = (low_K <= eff_K && eff_K < upper_K5);
            const bool K4_fit_L1 = (low_K <= eff_K && eff_K < upper_K4);
            const bool bd_big = (BD > 32);
            const bool ld_big = (LD > 32);
            const bool aligned_bd_mask
                    = brg->brgattr.bd_mask_level != 0 && brg->bdb % 4 == 0;
            if (LD % 80 == 0 && K5_fit_L1 && bd_big) {
                recalc_blocking_ext(
                        0, 16, 1, 5, true, false, brgemm_bd_loop_innermost);
            } else if (LD % 64 == 0 && K4_fit_L1 && bd_big) {
                recalc_blocking_ext(
                        0, 16, 1, 4, true, false, brgemm_bd_loop_innermost);
            } else if ((BD % 80 == 0 || aligned_bd_mask) && K5_fit_L1
                    && ld_big) {

                recalc_blocking_ext(
                        0, 16, 5, 1, false, true, brgemm_ld_loop_innermost);
            } else if ((BD % 64 == 0 || aligned_bd_mask) && K4_fit_L1
                    && ld_big) {
                recalc_blocking_ext(
                        16, 16, 4, 1, false, true, brgemm_ld_loop_innermost);
            }
        }
        // Tile decomposition for shapes with small dimensions
        // or dimensions with tails
        const bool weak_ldb = brg->ld_block < 8 || brg->ldb_tail > 0;
        const bool weak_bdb = brg->bd_block < 8 || brg->bdb_tail > 0;
        const bool ldb_tail_only = ldb_tail_16 && !bdb_block_tail;
        const bool bdb_tail_only = bdb_block_tail && !ldb_tail_16;
        if (ldb_tail_only && LD > 64 && brg->ld_block < 8) {
            recalc_blocking(0, 16, 2, 1);
        } else if (ldb_tail_only && weak_ldb && LD_R16 == 64) {
            recalc_blocking(0, 16, 1, 4);
        } else if (ldb_tail_only && weak_ldb && LD_R16 == 48) {
            recalc_blocking(0, 16, 1, 3);
        } else if (ldb_tail_only && weak_ldb && LD_R16 == 32) {
            recalc_blocking(0, 16, 2, 2);
        } else if (BD <= 16) {
            // Have to call recalc_blocking twice to calculate ldb
            recalc_blocking(BD, 16, 0, 0);
            const auto ld_block2 = nstl::min(
                    ldb_tail_16 ? ((brg->ldb > 4) ? 3 : 4) : 5, div_up(LD, 16));
            recalc_blocking(0, 0, 1, ld_block2);
        } else if (bdb_tail_only && weak_bdb && BD > 64) {
            recalc_blocking(16, 16, 1, 2);
        } else if (bdb_tail_only && weak_bdb && BD_R16 == 64) {
            recalc_blocking(16, 16, 4, 1);
        } else if (bdb_tail_only && weak_bdb && BD_R16 == 48) {
            recalc_blocking(16, 16, 3, 1);
        } else if (bdb_tail_only && weak_bdb && BD_R16 == 32
                && (LD % 32 == 0)) {
            recalc_blocking(16, 16, 2, 2);
        } else if (LD <= 16) {
            // Have to call recalc_blocking twice to calculate bdb
            // we can't use ld_block other than 16
            recalc_blocking(16, 16, 0, 0);
            const auto bd_block2 = nstl::min(
                    brg->bdb_tail ? (brg->bdb > 4 ? 3 : 4) : 5, div_up(BD, 16));
            recalc_blocking(0, 0, bd_block2, 1);
        } else if (bdb_block_tail && ldb_tail_16 && BD_R16 == 32 && LD_R16 == 32
                && (weak_ldb || weak_bdb)) {
            recalc_blocking(16, 16, 2, 2);
        }

        // The code below is a draft for the future optimization of interleave
        // stores and small number of iterations.
        // TODO: review and enable if needed
#if 0
        // if interleave stores and small number of iterations then
        // try to increase them
        const auto n_iterations = brg->bdb2 * brg->bdb2;
        if (brg->brgattr.use_interleave_stores && n_iterations < 4) {
            int k_it = div_up(4, n_iterations);
            if (brg->bdb2 > brg->ldb2)
                recalc_blocking(0, 0, div_up(brg->bdb2, k_it), 0);
            else
                recalc_blocking(0, 0, 0, div_up(brg->ldb2, k_it));
        }
#endif
    }

    if (brg->get_num_A_tiles() + brg->get_num_B_tiles() + brg->get_num_C_tiles()
            > brgemm_desc_t::AMX_TILES_NUM) {
        assert(!"brgemm internal error: invalid blocking");
        return status::runtime_error;
    }

    // check hints for blocking parameters
    recalc_blocking(brg->brgattr.hint_bd_block, brg->brgattr.hint_ld_block,
            brg->brgattr.hint_bd_block2 ? brg->brgattr.hint_bd_block2
                                        : brg->bd_block2,
            brg->brgattr.hint_ld_block2 ? brg->brgattr.hint_ld_block2
                                        : brg->ld_block2);

    if (brg->brgattr.hint_load_nt_A != brgemm_hint_nt_undef)
        brg->load_nt_A = (brg->brgattr.hint_load_nt_A == brgemm_hint_nt_true);
    if (brg->brgattr.hint_load_nt_B != brgemm_hint_nt_undef)
        brg->load_nt_B = (brg->brgattr.hint_load_nt_B == brgemm_hint_nt_true);

    // TODO: if rd_block calculated is very small then maybe it makes
    // sense to use 1x2 or 2x1 blocking with supporting rd_block
    // and rdb_tail
    const auto rd_block_step = brg->rd_block_step();
    const auto max_rd_block = brg->max_rd_block();
    if (brg->amx_may_extend_k()) {
        brg->rd_block = nstl::min(
                rnd_up(brg->reduce_dim, brg->rd_step), max_rd_block);
    } else {
        brg->rd_block = rd_block_step;
        for (int i = max_rd_block; i > 0; i -= rd_block_step) {
            if (brg->reduce_dim % i == 0) {
                brg->rd_block = i;
                break;
            }
        }
    }

    brg->rdb = brg->reduce_dim / brg->rd_block;
    brg->rdb_tail = brg->reduce_dim % brg->rd_block;

    // Remove these guards in the future (add tail processing by reduction
    // dimension)
    // TODO: these checks do not work for fp8-f16 and f16-fp8 cfgs
    if (!IMPLICATION(brg->rdb > 0 && brg->rdb_tail,
                brg->is_tf32 || brg->is_input_convert()
                        || brg->amx_wary_k_tail())) {
        return status::unimplemented;
    }

    if (!IMPLICATION((brg->rdb_tail
                             % ((brg->is_bf16_tmm || brg->is_f16_tmm) ? 2 : 4))
                        != 0,
                brg->is_tf32 || brg->is_input_convert()
                        || brg->amx_wary_k_tail())) {
        return status::unimplemented;
    }

    //TODO: check this condition
    brg->interleave_tilestores_ = brg->beta == 0
                    && (brg->brgattr.use_interleave_stores
                            && (brg->bd_block2 * brg->ld_block2 == 4)
                            && !brg->brgattr.var_bs)
            ? true
            : false;
    return status::success;
}

status_t brgemm_blocking_vmm(brgemm_desc_t *brg) {
    const auto L1 = platform::get_per_core_cache_size(1);

    const int simd_w = is_superset(brg->isa_impl, avx512_core) ? 16 : 8;
    brg->ld_block = simd_w;
    brg->ldb = brg->load_dim / brg->ld_block;
    brg->ldb_tail = brg->load_dim % brg->ld_block;

    const int max_vpad = nstl::max(
            brg->brgattr.max_top_vpad, brg->brgattr.max_bottom_vpad);

    // iterate ld_block2 starting from 4 to allow bd_block larger than
    // virtual padding
    int max_bcast_block {0}, min_bcast_block {0}, adj_ld_block2 {0};
    bool few_regs = utils::one_of(brg->isa_impl, avx2, avx2_vnni, avx2_vnni_2);
    bool hint_n_bcast_1_load
            = brg->brgattr.hint_loop_order == brgemm_lo_bl_1load;
    for (int try_ld_block2 = 4; try_ld_block2 > 0; --try_ld_block2) {
        adj_ld_block2 = calculate_ldb_params(brg, try_ld_block2);
        brg->n_bcast_1_load
                = (few_regs && adj_ld_block2 == 4) || hint_n_bcast_1_load;
        max_bcast_block = calculate_max_bcast_block(brg, adj_ld_block2);
        const auto bdb_tail = brg->bcast_dim % max_bcast_block;
        min_bcast_block = bdb_tail > 0 ? bdb_tail : max_bcast_block;
        if (min_bcast_block >= max_vpad) break;
    }
    // bcast block in brgemm kernel should be greater than virtual
    // padding to avoid possible functional issues
    if (min_bcast_block < max_vpad) return status::unimplemented;

    const int min_block = nstl::max(1, max_vpad);

    float best_bd_block_eff = 0.f;
    brg->bd_block = max_bcast_block;
    for (int bd_block = max_bcast_block; bd_block >= min_block; bd_block--) {
        const auto bd_block_disb = static_cast<float>(brg->bcast_dim)
                / rnd_up(brg->bcast_dim, bd_block);
        const auto brgemm_microkernel_eff
                = (static_cast<float>(adj_ld_block2) * bd_block)
                / (((adj_ld_block2) + bd_block) * max_bcast_block);
        const auto bd_block_eff = bd_block_disb * brgemm_microkernel_eff;

        float block_foot_print = static_cast<float>(brg->typesize_A) * bd_block
                * brg->reduce_dim;
        if (block_foot_print <= static_cast<float>(L1)
                && (bd_block_eff > best_bd_block_eff)) {
            brg->bd_block = bd_block;
            best_bd_block_eff = bd_block_eff;
        }
    }
    brg->bdb = brg->bcast_dim / brg->bd_block;
    brg->bdb_tail = brg->bcast_dim % brg->bd_block;

    const int rd_unroll = 4;
    const data_type_t rd_block_dt = get_mac_emu_data_type(
            brg->dt_a, brg->isa_impl, brg->isa_impl != avx2_vnni_2);
    if (rd_block_dt == dnnl_data_type_undef) return status::unimplemented;
    const int vnni_granularity = data_type_vnni_granularity(rd_block_dt);
    brg->rd_block = rd_unroll * vnni_granularity;
    brg->rdb = brg->reduce_dim / brg->rd_block;
    brg->rdb_tail = brg->reduce_dim % brg->rd_block;

    brg->is_M_tail = false;
    // avx2_vnni_2 kernel with xf16 data type requires blocked weights.
    if (brg->isa_impl == avx2_vnni_2 && brg->is_xf16()
            && brg->LDB % brg->ld_block > 0)
        return status::unimplemented;

    return status::success;
}

status_t brgemm_blocking(brgemm_desc_t *brg) {
    const data_type_t ld_step_compute_dt = get_mac_emu_data_type(
            brg->dt_b, brg->isa_impl, brg->isa_impl != avx2_vnni_2);
    brg->ld_step = brg->is_f16_b_non_amx_vnni()
            ? 2
            : data_type_vnni_granularity(ld_step_compute_dt);
    const data_type_t rd_step_compute_dt
            = get_mac_emu_data_type(brg->dt_b, brg->isa_impl);
    brg->rd_step = data_type_vnni_granularity(rd_step_compute_dt);

    set_isa_impl(brg);
    if (brg->isa_impl == isa_undef) return status::unimplemented;
    assert(!brg->is_dgmm); // should not be called from brdgmm
    if (brg->is_dgmm) return status::unimplemented;
    set_brg_vmm(brg);
    if (!(brg->is_tmm || brg->is_zmm || brg->is_ymm))
        return status::unimplemented;

    if (brg->is_tmm)
        CHECK(brgemm_blocking_tmm(brg));
    else
        CHECK(brgemm_blocking_vmm(brg));

    if (!IMPLICATION(brg->brgattr.LDB2 == 0, brg->load_dim <= brg->LDB))
        return status::invalid_arguments;

    brg->LDA2 = (brg->brgattr.LDA2 != 0) ? brg->brgattr.LDA2 : brg->LDA;
    brg->LDB2 = (brg->brgattr.LDB2 != 0) ? brg->brgattr.LDB2 : brg->LDB;
    brg->LDC2_M = (brg->brgattr.LDC2_M != 0) ? brg->brgattr.LDC2_M : brg->LDC;
    brg->LDC2_N
            = (brg->brgattr.LDC2_N != 0) ? brg->brgattr.LDC2_N : brg->ld_block;

    brg->is_blocked = (brg->LDA2 != brg->LDA || brg->LDB2 != brg->LDB
            || brg->LDC2_M != brg->LDC || brg->LDC2_N != brg->ld_block);

    if (!IMPLICATION(brg->is_blocked, brg->layout == brgemm_row_major))
        return status::invalid_arguments;

    return status::success;
}

status_t brdgmm_blocking(brgemm_desc_t *brg) {

    if (brg->isa_impl == isa_undef) return status::unimplemented;

    set_brg_vmm(brg); // Needed to dispatch into the right kernel later.
    const int max_vregs = isa_num_vregs(brg->isa_impl);

    const int simd_w = isa_max_vlen(brg->isa_impl) / brg->typesize_C;
    const bool is_avx2_vnni_2_xf16
            = brg->is_xf16() && brg->isa_impl == avx2_vnni_2;

    auto &M = brg->bcast_dim;
    auto &N = brg->load_dim;

    // In current implementation of dgmm, there is no reduce dim.
    auto &m_block1 = brg->bd_block;
    auto &nb_m_block1 = brg->bdb;
    auto &m_block1_tail = brg->bdb_tail;
    auto &m_block2 = brg->bd_block2;
    auto &nb_m_block2 = brg->bdb2;
    auto &m_block2_tail = brg->bdb2_tail;

    auto &n_block1 = brg->ld_block;
    auto &nb_n_block1 = brg->ldb;
    auto &n_block1_tail = brg->ldb_tail;
    auto &n_block2 = brg->ld_block2;
    auto &nb_n_block2 = brg->ldb2;
    auto &n_block2_tail = brg->ldb2_tail;

    // begin blocking
    // for avx2_vnni_2_xf16, instead of processing a n_block1 at once, it is
    // processed as even/odd pair.
    const int n_block1_num_steps = is_avx2_vnni_2_xf16 ? 2 : 1;
    n_block1 = n_block1_num_steps * simd_w;
    nb_n_block1 = div_up(N, n_block1);
    n_block1_tail = N % n_block1;

    const int max_n_block2_vmms = 4;
    const int max_n_block2 = max_n_block2_vmms / n_block1_num_steps;
    n_block2 = nstl::min(max_n_block2, nb_n_block1);

    const int aux_vregs
            = jit_brdgmm_kernel_base_t<Xbyak::Zmm>::get_aux_vmm_count(*brg);
    const int compute_vregs
            = jit_brdgmm_kernel_base_t<Xbyak::Zmm>::get_compute_vmm_count(*brg);
    const int bf16_emu_vregs = brg->is_bf16_emu * 4;
    const int postops_regs = brg->attr()
            ? injector::aux_vec_count(
                    brg->attr()->post_ops_, brg->isa_impl, true)
            : 0;

    const int max_acc_vmms = max_vregs
            - nstl::max(postops_regs,
                    nstl::max(compute_vregs + aux_vregs, bf16_emu_vregs));

    if (brg->brgattr.hint_bs_group > 1) {
        // Check if we can actually apply bs grouping
        const auto min_possible_m_block2
                = (max_acc_vmms / (2 * n_block1_num_steps)
                          - brg->brgattr.hint_bs_group + 1)
                / 2;
        if (min_possible_m_block2 < 1) brg->bs_group = 1;
    }

    if (brg->bs_group > 1) n_block2 = n_block2 % 2 == 0 ? 2 : 1;

    nb_n_block2 = div_up(nb_n_block1, n_block2);
    n_block2_tail = nb_n_block1 % n_block2;

    m_block1 = 1;
    nb_m_block1 = M / m_block1;
    m_block1_tail = M % m_block1;

    m_block2 = nstl::min(nb_m_block1,
            brg->bs_group > 1 ? (max_acc_vmms / (n_block2 * n_block1_num_steps)
                                        - brg->bs_group + 1)
                            / 2
                              : max_acc_vmms / (n_block2 * n_block1_num_steps));
    assert(m_block2 > 0);
    nb_m_block2 = div_up(nb_m_block1, m_block2);
    m_block2_tail = nb_m_block1 % m_block2;

    return status::success;
}

status_t init_brgemm_conf(brgemm_desc_t *brg, cpu_isa_t isa,
        brgemm_batch_kind_t type, impl::data_type_t dt_a,
        impl::data_type_t dt_b, brgemm_layout_t layout, float alpha, float beta,
        dim_t LDA, dim_t LDB, dim_t LDC, dim_t M, dim_t N, dim_t K,
        const brgemm_strides_t *strides, bool is_bf32, bool is_tf32) {

    init_common_conf(brg, type, alpha, beta, strides);

    brg->layout = layout;

    brg->dt_a = brg->is_row_major() ? dt_a : dt_b;
    brg->dt_b = brg->is_row_major() ? dt_b : dt_a;
    CHECK(init_kernel_datatype(brg, brg->dt_a, brg->dt_b));

    brg->dt_c = get_accum_datatype(brg);
    brg->dt_d = brg->dt_c;
    brg->dt_bias = brg->dt_c;

    brg->typesize_A = types::data_type_size(brg->dt_a);
    brg->typesize_B = types::data_type_size(brg->dt_b);
    brg->typesize_C = types::data_type_size(brg->dt_c);
    brg->typesize_D = types::data_type_size(brg->dt_d);

    brg->isa_user = isa;

    brg->is_tf32 = is_tf32
            && utils::one_of(brg->isa_user, isa_undef, avx10_2_512_amx_2)
            && mayiuse(avx10_2_512_amx_2);
    brg->is_bf32 = is_bf32
            && utils::one_of(brg->isa_user, isa_undef, avx512_core_amx)
            && mayiuse(avx512_core_amx);

    set_isa_impl(brg);
    brg->is_int8_tmm
            = brg->is_int8 && is_superset(brg->isa_impl, avx512_core_amx);
    brg->is_bf16_tmm
            = brg->is_bf16 && is_superset(brg->isa_impl, avx512_core_amx);
    brg->is_f16_tmm
            = brg->is_f16 && is_superset(brg->isa_impl, avx512_core_amx_fp16);
    brg->is_fp8_tmm
            = brg->is_fp8 && is_superset(brg->isa_impl, avx512_core_amx_fp16);

    brg->has_int8_vnni = isa_has_int8_vnni(brg->isa_impl);

    set_brg_vmm(brg); // TODO: Investigate if it is really needed here.
    brg->req_s8s8_compensation = brg->is_int8 && brg->dt_a == data_type::s8
            && !isa_has_s8s8(brg->isa_impl);

    brg->LDA = (brg->is_row_major()) ? static_cast<int>(LDA)
                                     : static_cast<int>(LDB);
    brg->is_runtime_lda = (brg->is_row_major()) ? is_runtime_value(LDA)
                                                : is_runtime_value(LDB);
    brg->LDB = (brg->is_row_major()) ? static_cast<int>(LDB)
                                     : static_cast<int>(LDA);
    brg->is_runtime_ldb = (brg->is_row_major()) ? is_runtime_value(LDB)
                                                : is_runtime_value(LDA);
    brg->LDC = static_cast<int>(LDC);
    brg->LDD = static_cast<int>(LDC);
    brg->is_runtime_ldc = brg->is_runtime_ldd = is_runtime_value(LDC);

    brg->bcast_dim
            = (brg->is_row_major()) ? static_cast<int>(M) : static_cast<int>(N);
    brg->load_dim
            = (brg->is_row_major()) ? static_cast<int>(N) : static_cast<int>(M);
    brg->reduce_dim = static_cast<int>(K);

    brg->bd_block2 = 0;
    brg->bdb2 = 0;
    brg->bdb2_tail = 0;

    return status::success;
}

status_t init_brdgmm_conf(brgemm_desc_t *brg, cpu_isa_t isa,
        brgemm_batch_kind_t type, impl::data_type_t dt_a,
        impl::data_type_t dt_b, brgemm_layout_t layout, float alpha, float beta,
        dim_t LDA, dim_t LDC, dim_t M, dim_t N,
        const brgemm_strides_t *strides) {

    init_common_conf(brg, type, alpha, beta, strides);

    brg->layout = layout;

    brg->dt_a = dt_a;
    brg->dt_b = dt_b;
    CHECK(init_kernel_datatype(brg, brg->dt_a, brg->dt_b));

    brg->dt_c = get_accum_datatype(brg);
    brg->dt_d = brg->dt_c;
    brg->dt_bias = brg->dt_c;

    brg->typesize_A = types::data_type_size(brg->dt_a);
    brg->typesize_B = types::data_type_size(brg->dt_b);
    brg->typesize_C = types::data_type_size(brg->dt_c);
    brg->typesize_D = types::data_type_size(brg->dt_d);

    brg->isa_user = isa;
    auto is_isa_ok = [&](cpu_isa_t isa) {
        return mayiuse(isa) && one_of(brg->isa_user, isa_undef, isa);
    };

    if (brg->is_f32) {
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(avx512_core),
                avx512_core, is_isa_ok(avx2), avx2);
    } else if (brg->is_bf16) {
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(avx512_core_bf16),
                avx512_core_bf16, is_isa_ok(avx2_vnni_2), avx2_vnni_2);
    } else if (brg->is_f16) {
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(avx512_core_fp16),
                avx512_core_fp16, is_isa_ok(avx2_vnni_2), avx2_vnni_2,
                is_isa_ok(avx10_2_512), avx10_2_512);
    } else if (brg->is_int8) {
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(avx10_2_512),
                avx10_2_512, is_isa_ok(avx512_core_vnni), avx512_core_vnni,
                is_isa_ok(avx2_vnni_2), avx2_vnni_2, is_isa_ok(avx2_vnni),
                avx2_vnni);
    }

    brg->req_s8s8_compensation = brg->is_int8 && brg->dt_a == data_type::s8
            && !isa_has_s8s8(brg->isa_impl);

    brg->is_dgmm = true;

    brg->LDA = static_cast<int>(LDA);
    brg->LDC = static_cast<int>(LDC);
    brg->LDD = static_cast<int>(LDC);

    brg->bcast_dim = M;
    brg->load_dim = N;

    return status::success;
}

} // namespace brgemm_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
