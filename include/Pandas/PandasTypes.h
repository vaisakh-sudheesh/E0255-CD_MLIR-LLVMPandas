//===- PandasTypes.h - Pandas dialect types -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PANDAS_PANDASTYPES_H
#define PANDAS_PANDASTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Pandas/PandasOpsTypes.h.inc"

#endif // PANDAS_PANDASTYPES_H
