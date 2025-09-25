//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_TypeInfo.hpp>
#include <gtest/gtest.h>

#include <type_traits>

namespace {

using Kokkos::Impl::TypeInfo;

struct Foo {};
using FooAlias = Foo;
enum Bar { BAR_0, BAR_1, BAR_2 };
union Baz {
  int i;
  float f;
};

[[maybe_unused]] auto func = [](int) {};  // < line 34
//                           ^  column 30
using Lambda = decltype(func);

TEST(defaultdevicetype, type_info) {
// clang-format off
#if defined(__NVCC__) && !defined(__CUDA_ARCH__)
// can't do much
// it looks like that there is 1st an EDG pass and then a host pass and they cannot both agree on what the type info is
#elif defined(__EDG__) || (defined(__NVCC__) && defined(__CUDA_ARCH__))
EXPECT_EQ(TypeInfo<Foo>::name()     , "<unnamed>::Foo");
EXPECT_EQ(TypeInfo<FooAlias>::name(), "<unnamed>::Foo");
EXPECT_EQ(TypeInfo<Bar>::name()     , "<unnamed>::Bar");
EXPECT_EQ(TypeInfo<Baz>::name()     , "<unnamed>::Baz");
EXPECT_EQ(TypeInfo<Lambda>::name()  , "lambda [](int)->void");
#elif defined(__clang__)
EXPECT_EQ(TypeInfo<Foo>::name()     , "(anonymous namespace)::Foo");
EXPECT_EQ(TypeInfo<FooAlias>::name(), "(anonymous namespace)::Foo");
EXPECT_EQ(TypeInfo<Bar>::name()     , "(anonymous namespace)::Bar");
EXPECT_EQ(TypeInfo<Baz>::name()     , "(anonymous namespace)::Baz");
EXPECT_EQ(TypeInfo<Lambda>::name()  , "(anonymous namespace)::(lambda at "  __FILE__  ":34:30)");
#elif defined(__GNUC__)
EXPECT_EQ(TypeInfo<Foo>::name()     , "{anonymous}::Foo");
EXPECT_EQ(TypeInfo<FooAlias>::name(), "{anonymous}::Foo");
EXPECT_EQ(TypeInfo<Bar>::name()     , "{anonymous}::Bar");
EXPECT_EQ(TypeInfo<Baz>::name()     , "{anonymous}::Baz");
EXPECT_EQ(TypeInfo<Lambda>::name()  , "{anonymous}::<lambda(int)>");
#elif defined(_MSC_VER)
EXPECT_EQ(TypeInfo<Foo>::name()     , "struct `anonymous-namespace'::Foo");
EXPECT_EQ(TypeInfo<FooAlias>::name(), "struct `anonymous-namespace'::Foo");
EXPECT_EQ(TypeInfo<Bar>::name()     , "enum `anonymous-namespace'::Bar");
EXPECT_EQ(TypeInfo<Baz>::name()     , "union `anonymous-namespace'::Baz");
// EXPECT_EQ(TypeInfo<Lambda>::name().starts_with("class `anonymous-namespace'::<lambda_"));
// underscore followed by some 32-bit hash that seems sensitive to the content of the current source code file
//EXPECT_EQ(TypeInfo<Lambda>::name().ends_with(">"));
#else
#error how did I ended up here?
#endif
}
// clang-format on

}  // namespace
