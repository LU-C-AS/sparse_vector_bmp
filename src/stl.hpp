
#pragma once
#include <algorithm>
#include <atomic>
#include <bit>
#include <bitset>
#include <cassert>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <forward_list>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
// #include <semaphore>
#include <set>
#include <shared_mutex>
// #include <source_location>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
#include <queue>
#include <deque>
#include <optional>

namespace sparse_vector_bmp {

// using ptr_t = char *;
// using const_ptr_t = const char *;
// using char_t = char;
// using SizeT = uint64_t;
// using uintptr_t = std::uintptr_t;

// template <typename T, typename Allocator = std::allocator<T>>
// using Vector = std::vector<T, Allocator>;

// template<typename T>
// using UniquePtr = std::unique_ptr<T>;

// template <typename T, typename... Args>
// inline UniquePtr<T> MakeUnique(Args && ...args) {
//     return std::make_unique<T>(std::forward<Args>(args)...);
// }

// template <typename T, typename... Args>
// inline UniquePtr<T> MakeUniqueForOverwrite(Args && ...args) {
//     return std::make_unique_for_overwrite<T>(std::forward<Args>(args)...);
// }

// template <typename T1, typename T2>
// using Pair = std::pair<T1, T2>;

// template <typename T, typename U>
// inline constexpr Pair<T, U> MakePair(T && first, U && second) {
//     return std::make_pair<T, U>(std::forward<T>(first), std::forward<U>(second));
// }

template <typename T1, typename T2>
using Pair = std::pair<T1, T2>;

template <typename... T>
using Tuple = std::tuple<T...>;

template <typename T, std::size_t N>
using Array = std::array<T, N>;

template <typename T, typename Allocator = std::allocator<T>>
using Vector = std::vector<T, Allocator>;

template <typename T>
using Span = std::span<T>;

template <typename T>
using Deque = std::deque<T>;

template <typename T>
using List = std::list<T>;

template <typename T>
using Queue = std::queue<T>;

template <typename S, typename T>
using Map = std::map<S, T>;

template <typename S, typename T>
using MultiMap = std::multimap<S, T>;

template <typename T>
using Set = std::set<T>;

template <typename T>
using Hash = std::hash<T>;

template <typename T>
struct EqualTo {
    bool operator()(const T &left, const T &right) const { return left == right; }
};

template <typename S, typename T, typename H = std::hash<S>, typename Eq = EqualTo<S>>
using HashMap = std::unordered_map<S, T, H, Eq>;

template <typename S, typename T, typename H = std::hash<S>>
using MultiHashMap = std::unordered_multimap<S, T, H>;

template <typename S, typename T = std::hash<S>, typename Eq = std::equal_to<S>>
using HashSet = std::unordered_set<S, T, Eq>;

template <typename T>
using MaxHeap = std::priority_queue<T>;

template <typename T, typename C>
using Heap = std::priority_queue<T, std::vector<T>, C>;

template <typename T>
using Optional = std::optional<T>;
constexpr std::nullopt_t None = std::nullopt;

using NoneType = std::nullopt_t;

// String

using String = std::basic_string<char>;

inline bool IsEqual(const String &s1, const String &s2) { return s1 == s2; }

inline bool IsEqual(const String &s1, const char *s2) { return s1 == s2; }

inline String TrimPath(const String &path) {
    const auto pos = path.find("/src/");
    if (pos == String::npos)
        return path;
    return path.substr(pos + 1);
}

inline String TrimString(const String &s) {
    int len = s.length();
    int i = 0;

    while (i < len && isspace(s[i])) {
        i++;
    }

    while (len > i && isspace(s[len - 1])) {
        len--;
    }

    if (i == len) {
        return "";
    }

    String ss = s.substr(i, len - i);
    return ss;
}

// std::vector<std::string> SplitStrByComma(String str) {
//     std::vector<std::string> tokens;
//     for (const auto &token : str | std::views::split(',')) {
//         tokens.emplace_back(token.begin(), token.end());
//     }

//     for (auto &s : tokens) {
//         s = TrimString(s);
//     }

//     return tokens;
// }

void ToUpper(String & str) { std::transform(str.begin(), str.end(), str.begin(), ::toupper); }

int ToUpper(int c) { return ::toupper(c); }

void ToLower(String & str) { std::transform(str.begin(), str.end(), str.begin(), ::tolower); }

int ToLower(int c) { return ::tolower(c); }

inline void StringToLower(String & str) {
    std::transform(str.begin(), str.end(), str.begin(), [](const auto c) { return std::tolower(c); });
}

template <class BidirIteratorType>
BidirIteratorType Prev(BidirIteratorType it, typename std::iterator_traits<BidirIteratorType>::difference_type n = 1) {
    std::advance(it, -n);
    return it;
}

template <class BidirIteratorType>
BidirIteratorType Next(BidirIteratorType it, typename std::iterator_traits<BidirIteratorType>::difference_type n = 1) {
    std::advance(it, n);
    return it;
}

// Primitives

using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using idx_t = u64;

using f32 = float;
using f64 = double;

using offset_t = int64_t;

using ptr_t = char *;
using const_ptr_t = const char *;
using char_t = char;
using SizeT = u64;
using uintptr_t = std::uintptr_t;

// Transactions
using TxnTimeStamp = uint64_t;
using TransactionID = uint64_t;

// Entry
using SegmentID = uint32_t;
using ChunkID = uint32_t;
using BlockID = uint16_t;
using ColumnID = uint64_t;

// Related to entry
using BlockOffset = uint16_t;
using SegmentOffset = uint32_t;

// Concurrency
// using ThreadPool = ctpl::thread_pool;

using Thread = std::thread;

using atomic_u32 = std::atomic_uint32_t;
using atomic_u64 = std::atomic_uint64_t;
using ai64 = std::atomic_int64_t;
using aptr = std::atomic_uintptr_t;
using atomic_bool = std::atomic_bool;

template <typename T>
using Atomic = std::atomic<T>;

using std::atomic_compare_exchange_strong;
using std::atomic_store;

// Smart ptr
template <typename T>
using SharedPtr = std::shared_ptr<T>;

template <typename T>
using WeakPtr = std::weak_ptr<T>;

template <typename T, typename... Args>
inline SharedPtr<T> MakeShared(Args && ...args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

template <typename T>
using UniquePtr = std::unique_ptr<T>;

template <typename T, typename... Args>
inline UniquePtr<T> MakeUnique(Args && ...args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

// template <typename T, typename... Args>
// inline UniquePtr<T> MakeUniqueForOverwrite(Args && ...args) {
//     return std::make_unique_for_overwrite<T>(std::forward<Args>(args)...);
// }

template <typename T, typename U>
inline constexpr Pair<T, U> MakePair(T && first, U && second) {
    return std::make_pair<T, U>(std::forward<T>(first), std::forward<U>(second));
}

// template <typename T>
// inline constexpr Optional<T> MakeOptional(T && value) {
//     return std::make_optional<T>(std::forward<T>(value));
// }

// Chrono
using Clock = std::chrono::high_resolution_clock;

template <typename T>
using TimePoint = std::chrono::time_point<T, std::chrono::nanoseconds>;

using NanoSeconds = std::chrono::nanoseconds;
using MicroSeconds = std::chrono::microseconds;
using MilliSeconds = std::chrono::milliseconds;
using Seconds = std::chrono::seconds;

inline NanoSeconds ElapsedFromStart(const TimePoint<Clock> &end, const TimePoint<Clock> &start) { return end - start; }

template <typename T>
T ChronoCast(const NanoSeconds &nano_seconds) {
    return std::chrono::duration_cast<T>(nano_seconds);
}

// // IsStandLayout
// template <typename T>
// concept IsStandLayout = std::is_standard_layout_v<T>;

// // Stringstream
// using IStringStream = std::istringstream;
// using OStringStream = std::ostringstream;

// // Dir
// using Path = std::filesystem::path;
// using DirEntry = std::filesystem::directory_entry;

// inline Vector<String> GetFilesFromDir(const String &path) {
//     Vector<String> result;
//     for (auto &i : std::filesystem::directory_iterator(path)) {
//         result.emplace_back(i.path().string());
//     }
//     return result;
// }

// typeid
//    using TypeID = std::typeid();

// std::function
//    template<typename R, typename... Ts>
//    using std::function = std::function<R, Ts>;

// SharedPtr
template <typename T>
using EnableSharedFromThis = std::enable_shared_from_this<T>;

template <typename II, typename OI>
OI Copy(II first, II last, OI d_first) {
    return std::copy(first, last, d_first);
}

template <typename T1, typename T2>
struct CompareByFirst {
    using P = std::pair<T1, T2>;

    bool operator()(const P &lhs, const P &rhs) const { return lhs.first < rhs.first; }
};

template <typename T1, typename T2>
struct CompareByFirstReverse {
    using P = std::pair<T1, T2>;

    bool operator()(const P &lhs, const P &rhs) const { return lhs.first > rhs.first; }
};
}