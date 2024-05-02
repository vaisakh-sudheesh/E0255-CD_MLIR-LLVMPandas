//===- PandasTypes.cpp - Pandas dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Pandas/PandasTypes.h"

#include "Pandas/PandasDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::pandas;

#define GET_TYPEDEF_CLASSES
#include "Pandas/PandasOpsTypes.cpp.inc"

void PandasDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Pandas/PandasOpsTypes.cpp.inc"
      >();
}
