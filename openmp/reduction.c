// gcc -O3 -mavx512f -fopenmp -march=native -flto reduction.c -o reduction
// author : 2300012929 yinjirnun (with deepseek)

#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef union {
  __m512i vec;
  long long arr[8]; // AVX-512可以容纳8个64位整数
} avx_long8;

#pragma omp declare reduction(avx512_add:avx_long8                             \
                              : omp_out.vec =                                  \
                                    _mm512_add_epi64(omp_out.vec, omp_in.vec)) \
    initializer(omp_priv = {.vec = _mm512_setzero_si512()})

void avx512_reduction(const long long *data, size_t size, long long *result) {
  avx_long8 sum = {.vec = _mm512_setzero_si512()};

#pragma omp parallel for reduction(avx512_add : sum)
  for (size_t i = 0; i < size / 8 * 8; i += 8) {
    __m512i chunk = _mm512_loadu_si512((const __m512i *)(data + i));
    sum.vec = _mm512_add_epi64(sum.vec, chunk);
  }

  *result = 0;
  for (int i = 0; i < 8; i++) {
    *result += sum.arr[i];
  }

  for (size_t i = size / 8 * 8; i < size; i++) {
    *result += data[i];
  }
}

void simple_reduction(const long long *data, size_t size, long long *result) {
  *result = 0;
  for (size_t i = 0; i < size; i++) {
    *result += data[i];
  }
}

void benchmark(const char *name,
               void (*func)(const long long *, size_t, long long *),
               const long long *data, size_t size, long long *result) {
  struct timespec start, end;

  // warmup
  func(data, size, result);

  clock_gettime(CLOCK_MONOTONIC, &start);

  for (int i = 0; i < 10; i++) {
    func(data, size, result);
  }

  clock_gettime(CLOCK_MONOTONIC, &end);

  double time_taken = (end.tv_sec - start.tv_sec) * 1e9;
  time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;
  time_taken /= 10; // 计算平均时间

  printf("%s:\n", name);
  printf("  Result: %lld\n", *result);
  printf("  Time: %.6f seconds\n", time_taken);
  printf("  Throughput: %.2f MB/s\n",
         (size * sizeof(long long)) / (time_taken * 1024 * 1024));
}

int main() {
  size_t n = 1e8;
  long long *a = malloc(n * sizeof(long long));
  if (a == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }

  unsigned int seed = time(NULL) ^ omp_get_thread_num();

#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    a[i] = rand_r(&seed) % 10 + 1;
  }

  long long result_avx, result_simple;

  printf("Benchmarking with %zu elements (%zu MB)\n", n,
         n * sizeof(long long) / (1024 * 1024));

  benchmark("AVX-512 Reduction", avx512_reduction, a, n, &result_avx);
  benchmark("Simple Reduction", simple_reduction, a, n, &result_simple);

  if (result_avx != result_simple) {
    printf("\nERROR: Results don't match! (%lld vs %lld)\n", result_avx,
           result_simple);
  } else {
    printf("\nResults verified: %lld\n", result_avx);
  }

  free(a);
  return 0;
}