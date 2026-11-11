// MLX v0.32.4 builds the JIT source of the head-dim-split attention kernel with
// one template argument too many (ml-explore/mlx issue 4832).
// `attention_nax_dsplit` takes no value head dimension, so the Metal compiler
// rejects the instantiation and MLX throws on the first fused attention that
// takes this kernel: head dim 256 or 512, 1024 or more query rows, a NAX GPU.
//
// This wrapper builds upstream's file with that one function renamed, then
// defines the function with the argument list each kernel declares. Delete this
// file and its Package.swift exclude when the vendored MLX has the fix.

#include "mlx/backend/metal/kernels.h"

#define get_steel_attention_nax_kernel get_steel_attention_nax_kernel_mlx_v0_32_4
#include "../mlx/mlx/backend/metal/jit_kernels.cpp"
#undef get_steel_attention_nax_kernel

namespace mlx::core {

MTL::ComputePipelineState* get_steel_attention_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& q,
    int bq,
    int bk,
    int bd,
    int bv,
    int wm,
    int wn,
    const array& m,
    bool split_d) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::string kernel_source;
    concatenate(
        kernel_source,
        metal::utils(),
        metal::steel_attention_nax(),
        split_d
            ? get_template_definition(
                  lib_name,
                  "attention_nax_dsplit",
                  get_type_string(q.dtype()),
                  bq,
                  bk,
                  bd,
                  wm,
                  wn,
                  get_type_string(m.dtype()))
            : get_template_definition(
                  lib_name,
                  "attention_nax",
                  get_type_string(q.dtype()),
                  bq,
                  bk,
                  bd,
                  bv,
                  wm,
                  wn,
                  get_type_string(m.dtype())));
    return kernel_source;
  });
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

} // namespace mlx::core
