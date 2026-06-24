/* Copyright 2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

           http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifdef WIN32
#include <tests/main.h>
#endif

#include <fast_npp.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <npp.h>
#include <nppi_color_conversion.h>
#include <vector>
#include <random>
#include <cstdlib>
#include <cstdio>

// RGBToGray: validated against nppiRGBToGray within +/-1 LSB (documented float
// rounding on exact-.5 ties; NPP's internal FMA order is not bit-replicable).
// The test asserts max abs diff <= 1 AND reports the exact-match percentage.
static int testRGBToGray8u(int W, int H) {
    const size_t N = (size_t)W * H;
    std::vector<uchar3> h(N);
    std::vector<unsigned char> ref(N), got(N);
    std::mt19937 r(7);
    std::uniform_int_distribution<int> di(0, 255);
    for (auto& v : h) { v.x = di(r); v.y = di(r); v.z = di(r); }

    fk::Ptr2D<uchar3> src(W, H);
    fk::Ptr2D<uchar> dst(W, H);
    const int sp = (int)src.ptr().dims.pitch;
    const int dp = (int)dst.ptr().dims.pitch;
    cudaMemcpy2D(src.ptr().data, sp, h.data(), W * 3, W * 3, H, cudaMemcpyHostToDevice);

    NppStreamContext ctx{};
    ctx.hStream = 0;
    cudaGetDevice(&ctx.nCudaDeviceId);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, ctx.nCudaDeviceId);
    ctx.nMultiProcessorCount = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    ctx.nCudaDevAttrComputeCapabilityMajor = prop.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
    cudaStreamGetFlags(ctx.hStream, &ctx.nStreamFlags);

    Npp8u* dref = nullptr;
    cudaMalloc(&dref, (size_t)dp * H);
    nppiRGBToGray_8u_C3C1R_Ctx((const Npp8u*)src.ptr().data, sp, dref, dp, NppiSize{ W, H }, ctx);
    cudaStreamSynchronize(ctx.hStream);
    cudaMemcpy2D(ref.data(), W, dref, dp, W, H, cudaMemcpyDeviceToHost);

    fk::Stream stream(ctx.hStream);
    fk::executeOperations<fk::TransformDPP<>>(stream,
        fk::PerThreadRead<fk::ND::_2D, uchar3>::build(src),
        fastNPP::RGBToGray_8u_C3C1R_Ctx(),
        fk::PerThreadWrite<fk::ND::_2D, uchar>::build(dst));
    stream.sync();
    cudaMemcpy2D(got.data(), W, dst.ptr().data, dp, W, H, cudaMemcpyDeviceToHost);

    int maxDiff = 0;
    size_t exact = 0;
    for (size_t i = 0; i < N; ++i) {
        const int d = std::abs((int)ref[i] - (int)got[i]);
        if (d > maxDiff) maxDiff = d;
        if (d == 0) ++exact;
    }
    cudaFree(dref);

    const double pct = 100.0 * (double)exact / (double)N;
    const bool pass = (maxDiff <= 1);
    printf("[%s] RGBToGray_8u_C3C1R %dx%d: maxDiff=%d exact=%.4f%% (tol +/-1 LSB)\n",
           pass ? "PASS" : "FAIL", W, H, maxDiff, pct);
    return pass ? 0 : 1;
}

int launch() {
    int bad = 0;
    bad += testRGBToGray8u(1920, 1080);
    bad += testRGBToGray8u(641, 480);
    printf("%s\n", bad == 0 ? "ALL PASS" : "FAILURES DETECTED");
    return bad == 0 ? 0 : -1;
}
