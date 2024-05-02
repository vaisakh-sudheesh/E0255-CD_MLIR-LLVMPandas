//===- PandasDialect.cpp - Pandas dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Pandas/PandasDialect.h"
#include "Pandas/PandasOps.h"
#include "Pandas/PandasTypes.h"

using namespace mlir;
using namespace mlir::pandas;

#include "Pandas/PandasOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Pandas dialect.
//===----------------------------------------------------------------------===//

void PandasDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Pandas/PandasOps.cpp.inc"
      >();
  registerTypes();
}
