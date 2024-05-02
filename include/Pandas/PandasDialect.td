//===- PandasDialect.td - Pandas dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PANDAS_DIALECT
#define PANDAS_DIALECT

include "mlir/IR/OpBase.td"

include "mlir/Dialect/LLVMIR/LLVMDialect.td"

//===----------------------------------------------------------------------===//
// Pandas dialect definition.
//===----------------------------------------------------------------------===//

def Pandas_Dialect : Dialect {
    let name = "pandas";
    let summary = "A pandas out-of-tree MLIR dialect.";
    let cppNamespace = "::mlir::pandas";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}



#endif // PANDAS_DIALECT
