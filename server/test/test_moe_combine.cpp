// Model-neutral correctness coverage for the fused MoE route-weight and
// expert-reduction primitive used by DS4, Laguna, and common hybrid graphs.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

int main() {
    constexpr int n_embd = 257;
    constexpr int n_used = 6;
    constexpr int n_tokens = 5;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "GPU backend unavailable\n");
        return 1;
    }

    ggml_init_params params{};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        return 1;
    }

    ggml_tensor * experts = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, n_embd, n_used, n_tokens);
    ggml_tensor * weights = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, n_used, n_tokens);
    ggml_set_input(experts);
    ggml_set_input(weights);
    ggml_tensor * output = ggml_moe_combine(ctx, experts, weights);
    ggml_set_output(output);

    bool ok = output->type == GGML_TYPE_F32 &&
              output->ne[0] == n_embd && output->ne[1] == n_tokens &&
              output->ne[2] == 1 && output->ne[3] == 1;

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ok = ok && allocator && ggml_gallocr_alloc_graph(allocator, graph);

    std::vector<float> expert_data(
        (size_t) n_embd * n_used * n_tokens);
    std::vector<float> weight_data((size_t) n_used * n_tokens);
    std::vector<float> expected((size_t) n_embd * n_tokens);
    std::vector<float> actual(expected.size());

    for (int t = 0; t < n_tokens; ++t) {
        for (int e = 0; e < n_used; ++e) {
            const size_t route = (size_t) t * n_used + e;
            weight_data[route] = ((e + 2 * t) % 5 == 0)
                ? 0.0f : 0.05f * (float) (e + 1);
            for (int h = 0; h < n_embd; ++h) {
                const size_t index = route * n_embd + h;
                expert_data[index] = weight_data[route] == 0.0f
                    ? std::numeric_limits<float>::quiet_NaN()
                    : 0.001f * (float) ((h % 29) - 14) +
                      0.01f * (float) (e - t);
            }
        }
    }

    for (int t = 0; t < n_tokens; ++t) {
        for (int h = 0; h < n_embd; ++h) {
            float sum = 0.0f;
            for (int e = 0; e < n_used; ++e) {
                const size_t route = (size_t) t * n_used + e;
                const float weight = weight_data[route];
                if (weight == 0.0f) {
                    if (e == 0) sum = 0.0f;
                    continue;
                }
                const float product = expert_data[route * n_embd + h] * weight;
                sum = e == 0 ? product : sum + product;
            }
            expected[(size_t) t * n_embd + h] = sum;
        }
    }

    if (ok) {
        ggml_backend_tensor_set(
            experts, expert_data.data(), 0, expert_data.size() * sizeof(float));
        ggml_backend_tensor_set(
            weights, weight_data.data(), 0, weight_data.size() * sizeof(float));
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        ggml_backend_tensor_get(
            output, actual.data(), 0, actual.size() * sizeof(float));
        for (size_t i = 0; i < actual.size(); ++i) {
            const float tolerance = 1.0e-6f *
                std::max(1.0f, std::abs(expected[i]));
            if (!std::isfinite(actual[i]) ||
                std::abs(actual[i] - expected[i]) > tolerance) {
                std::fprintf(
                    stderr, "mismatch at %zu actual=%g expected=%g\n",
                    i, actual[i], expected[i]);
                ok = false;
                break;
            }
        }
    }

    if (ok) {
        std::printf(
            "moe combine embd=%d experts=%d tokens=%d PASS\n",
            n_embd, n_used, n_tokens);
    }

    if (allocator) ggml_gallocr_free(allocator);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
