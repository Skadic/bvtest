#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <ranges>
#include <vector>

#include <bm64.h>
#include <dynamic/dynamic.hpp>
#include <la_vector.hpp>
#include <sdsl/sd_vector.hpp>

#define PROJECT_NAME "bvtest"

#define QUERIES static_cast<size_t>(1000000)
#define N       static_cast<size_t>(10000000)
#define FILL    0.1
static_assert(0 <= FILL && FILL <= 1, "fill rate must be between 0 and 1");

using Time  = timespec;
using Clock = clockid_t;

void print_result_line(std::string ds,
                       double      rank_ms,
                       double      select_ms,
                       size_t      bytes,
                       size_t      rank_checksum,
                       size_t      select_checksum) {
    std::cout << "RESULT ds=" << ds << " num_queries=" << QUERIES << " num_bits=" << N << " fill_rate=" << FILL
              << " rank=" << rank_ms << " select=" << select_ms << " space=" << bytes
              << " rank_checksum=" << rank_checksum << " select_checksum=" << select_checksum << std::endl;
}

auto duration(Time a, Time b) -> double { return 1000 * (a.tv_sec - b.tv_sec) + 1e-6 * (a.tv_nsec - b.tv_nsec); }

auto gen_bits() -> std::vector<bool> {
    auto gen = std::bind(std::uniform_real_distribution<double>(0.0, 1.0), std::default_random_engine());

    std::vector<bool> bv;
    bv.reserve(N);
    for (size_t i = 0; i < N; i++) {
        bv.push_back(gen() <= FILL);
    }
    return bv;
}

void test_bm(const std::vector<bool> &bits) {
    auto   ds = "bitmagic";
    double rank_ms, select_ms;
    size_t rank_checksum = 0, select_checksum = 0, bytes;

    bm::bvector<>                bv;
    bm::bvector<>::rs_index_type rs;
    bv.resize(N);

    for (size_t i = 0; i < N; i++) {
        bv.set(i, bits[i]);
    }
    bv.optimize();
    bv.freeze();
    bv.build_rs_index(&rs);

    {
        auto gen = std::bind(std::uniform_int_distribution<size_t>(0, N - 1), std::default_random_engine());
        Time now, then;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &now);
        for (size_t i = 0; i < QUERIES; i++) {
            rank_checksum += bv.rank(gen(), rs);
        }
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &then);
        rank_ms = duration(then, now);
    }
    {
        const auto num_ones = bv.rank(bits.size(), rs) - 1;
        auto       gen = std::bind(std::uniform_int_distribution<size_t>(1, num_ones), std::default_random_engine());
        Time       now, then;

        // Select Queries
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &now);
        for (size_t i = 0; i < QUERIES; i++) {
            bm::bvector<>::size_type s;
            bv.select(gen(), s, rs);
            select_checksum += s;
        }
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &then);
        select_ms = duration(then, now);
    }

    bm::bvector<>::statistics stats;
    bv.calc_stat(&stats);
    bytes = stats.memory_used;
    print_result_line(ds, rank_ms, select_ms, bytes, rank_checksum, select_checksum);
}

void test_dyn_succ(const std::vector<bool> &bits) {
    auto   ds = "dynsucc";
    double rank_ms, select_ms;
    size_t rank_checksum = 0, select_checksum = 0, bytes;

    dyn::succinct_bitvector<dyn::succinct_spsi> bv;
    for (size_t i = 0; i < N; i++) {
        bv.push_back(bits[i]);
    }

    {
        auto gen = std::bind(std::uniform_int_distribution<size_t>(0, N - 1), std::default_random_engine());
        Time now, then;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &now);
        for (size_t i = 0; i < QUERIES; i++) {
            rank_checksum += bv.rank(gen() + 1);
        }
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &then);
        rank_ms = duration(then, now);
    }

    {
        const auto num_ones = bv.rank(bits.size()) - 1;
        auto       gen = std::bind(std::uniform_int_distribution<size_t>(1, num_ones), std::default_random_engine());
        Time       now, then;

        // Select Queries
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &now);
        for (size_t i = 0; i < QUERIES; i++) {
            select_checksum += bv.select(gen() - 1);
        }
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &then);
        select_ms = duration(then, now);
    }
    bytes = bv.bit_size() / 8;
    print_result_line(ds, rank_ms, select_ms, bytes, rank_checksum, select_checksum);
}

void test_sdvec(const std::vector<bool> &bits) {
    auto            ds = "sdvector";
    double          rank_ms, select_ms;
    size_t          rank_checksum = 0, select_checksum = 0, bytes = 0;

    sdsl::sd_vector bv;
    {
        size_t one_count = 0;
        for (bool b : bits | std::views::filter([](bool i) { return i; })) {
            one_count++;
        }
        std::cout << "n: " << bits.size() << ", m: " << one_count << std::endl;

        sdsl::sd_vector_builder svb(bits.size(), one_count);
        for (size_t i = 0; i < bits.size(); i++) {
            if (bits[i]) {
                svb.set(i);
            }
        }
        bv = sdsl::sd_vector(svb);
    }

    sdsl::rank_support_sd rnk(&bv);
    sdsl::select_support_sd sel(&bv);

    {
        auto gen = std::bind(std::uniform_int_distribution<size_t>(0, N - 1), std::default_random_engine());
        Time now, then;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &now);
        for (size_t i = 0; i < QUERIES; i++) {
            rank_checksum += rnk.rank(gen() + 1);
        }
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &then);
        rank_ms = duration(then, now);
    }

    {
        const auto num_ones = rnk.rank(bits.size()) - 1;
        auto       gen = std::bind(std::uniform_int_distribution<size_t>(1, num_ones), std::default_random_engine());
        Time       now, then;

        // Select Queries
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &now);
        for (size_t i = 0; i < QUERIES; i++) {
            select_checksum += sel.select(gen());
        }
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &then);
        select_ms = duration(then, now);
    }

    print_result_line(ds, rank_ms, select_ms, bytes, rank_checksum, select_checksum);
}

void test_la(const std::vector<bool> &bits) {
    auto                ds = "la_vector";
    double              rank_ms, select_ms;
    size_t              rank_checksum = 0, select_checksum = 0, bytes;
    std::vector<size_t> a;
    a.reserve(std::min((size_t) (1.2 * FILL * N), N));
    for (size_t i = 0; i < bits.size(); i++) {
        if (bits[i]) {
            a.push_back(i);
        }
    }

    la_vector_opt<size_t> bv(a);
    {
        auto gen = std::bind(std::uniform_int_distribution<size_t>(0, N - 1), std::default_random_engine());
        Time now, then;

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &now);
        for (size_t i = 0; i < QUERIES; i++) {
            rank_checksum += bv.rank(gen() + 1);
        }
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &then);
        rank_ms = duration(then, now);
    }

    {
        const auto num_ones = bv.rank(bits.size()) - 1;
        auto       gen = std::bind(std::uniform_int_distribution<size_t>(1, num_ones), std::default_random_engine());
        Time       now, then;

        // Select Queries
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &now);
        for (size_t i = 0; i < QUERIES; i++) {
            select_checksum += bv.select(gen());
        }
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &then);
        select_ms = duration(then, now);
    }

    bytes = bv.size_in_bytes();
    print_result_line(ds, rank_ms, select_ms, bytes, rank_checksum, select_checksum);
}

auto main(int argc, char **argv) -> int {
    if (argc != 1) {
        std::cout << argv[0] << "takes no arguments.\n";
        return 1;
    }
    const auto bits = gen_bits();
    std::cout << "Starting benchmarks, " << QUERIES << " each..." << std::endl;

    test_sdvec(bits);
    test_dyn_succ(bits);
    test_bm(bits);
    test_la(bits);
    return 0;
}
