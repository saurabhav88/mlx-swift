import Foundation
import MLX
import XCTest

/// Causal attention at the shapes that select the head-dim-split kernel on a NAX GPU: 1024 query
/// rows, 16-bit floats, head dim 256 or 512. MLX v0.32.4 could not build that kernel's JIT source
/// (ml-explore/mlx issue 4832), so the evaluation threw. `mlx-conditional/jit_kernels_conditional.cpp`
/// repairs it. Other GPUs take another kernel for these shapes and must give the same answer.
class AttentionHeadDimSplitTests: XCTestCase {

    /// Largest absolute difference from the same attention computed in float32 with plain operations.
    private func maxDifference(heads: Int, kvHeads: Int, headDim: Int, length: Int) -> Float {
        MLXRandom.seed(11)
        let q = MLXRandom.normal([1, heads, length, headDim]).asType(.bfloat16)
        let k = MLXRandom.normal([1, kvHeads, length, headDim]).asType(.bfloat16)
        let v = MLXRandom.normal([1, kvHeads, length, headDim]).asType(.bfloat16)
        let scale = 1 / Float(headDim).squareRoot()

        let fused = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .causal)

        let group = heads / kvHeads
        let k32 = repeated(k.asType(.float32), count: group, axis: 1)
        let v32 = repeated(v.asType(.float32), count: group, axis: 1)
        var scores = (q.asType(.float32) * scale).matmul(k32.transposed(0, 1, 3, 2))
        scores = scores + triu(full([length, length], values: -Float.infinity), k: 1)
        let reference = softmax(scores, axis: -1).matmul(v32)

        XCTAssertEqual(fused.shape, reference.shape)
        let difference = abs(fused.asType(.float32) - reference).max().item(Float.self)
        // The bound below is a coarse check of a 16-bit kernel, not a precision claim.
        print("AttentionHeadDimSplit headDim=\(headDim) maxDifference=\(difference)")
        return difference
    }

    func testHeadDim256CausalAt1024Rows() {
        // The shape of a Qwen 3.5 prefill of 1024 tokens: 16 query heads over 4 key heads.
        let difference = maxDifference(heads: 16, kvHeads: 4, headDim: 256, length: 1024)
        XCTAssertLessThan(difference, 0.05)
    }

    func testHeadDim512CausalAt1024Rows() {
        // Head dim 512 needs 1024 query blocks of 32 rows to select the kernel: 32 heads.
        let difference = maxDifference(heads: 32, kvHeads: 8, headDim: 512, length: 1024)
        XCTAssertLessThan(difference, 0.05)
    }
}
