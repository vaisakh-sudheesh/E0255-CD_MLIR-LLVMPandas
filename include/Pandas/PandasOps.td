//===- PandasOps.td - Pandas dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PANDAS_OPS
#define PANDAS_OPS

include "Pandas/PandasTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"


//===----------------------------------------------------------------------===//
// Base pandas operation definition.
//===----------------------------------------------------------------------===//

class Pandas_Op<string mnemonic, list<Trait> traits = []> :
        Op<Pandas_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// ReadCSVOp
//===----------------------------------------------------------------------===//
def ReadCSVOp : Pandas_Op<"read_csv"> {
    let summary   = "read csv data to dataframe";

    let arguments = (ins StrAttr:$tablename, StrAttr:$filename);
    let results   = (outs LLVMPointerType:$dfref);

    // Specialize assembly printing and parsing using a declarative format.
    let assemblyFormat = [{
    `(` $tablename `,` $filename `)` attr-dict `:` type($dfref)
    }];

    let builders = [
        OpBuilder<(ins "std::string":$tablename ,"std::string":$filename)>
    ];
}

//===----------------------------------------------------------------------===//
// PrintDFOp
//===----------------------------------------------------------------------===//
def PrintDFOp : Pandas_Op<"print_df"> {
    let summary   = "print rows in a data frame";

    let arguments = (ins LLVMPointerType:$tablename, I32:$numrows);
    let results   = (outs LLVMPointerType:$dfref);

    let assemblyFormat = [{
    `(` $tablename `,` $numrows `)` attr-dict `:` `(`type($tablename) `,` type($numrows) `)` type($dfref)
    }];

    let builders = [
        OpBuilder<(ins "std::string":$tablename)>
    ];
}

//===----------------------------------------------------------------------===//
// MergeOnDFOp
//===----------------------------------------------------------------------===//
def MergeOnDFOp : Pandas_Op<"merge_on"> {
    let summary   = "Performs merge operation on shared column";

    let arguments = (ins LLVMPointerType:$srctable1, LLVMPointerType:$srctable2, StrAttr:$colname);
    let results   = (outs LLVMPointerType:$dfref);

    let assemblyFormat = [{
    `(` $srctable1 `:` type($srctable1) `,` $srctable2 `:` type($srctable2) `,` $colname `)` attr-dict type($dfref)
    }];

    let builders = [
        OpBuilder<(ins "std::string":$tablename)>
    ];
}

//===----------------------------------------------------------------------===//
// MergeLRDFOp
//===----------------------------------------------------------------------===//
def MergeLRDFOp : Pandas_Op<"merge_lr"> {
    let summary   = "Performs merge operation with left-on and right-on";

    let arguments = (ins LLVMPointerType:$srctable1, LLVMPointerType:$srctable2, StrAttr:$left_on_col, StrAttr:$righton_col);
    let results   = (outs LLVMPointerType:$dfref);

    let assemblyFormat = [{
    `(` $srctable1 `:` type($srctable1) `,` $srctable2 `:` type($srctable2) `,` $left_on_col `,` $righton_col `)` attr-dict type($dfref)
    }];

    let builders = [
        OpBuilder<(ins "std::string":$tablename)>
    ];
}

//===----------------------------------------------------------------------===//
// MergeFullDFOp
//===----------------------------------------------------------------------===//
def MergeFullDFOp : Pandas_Op<"merge_full"> {
    let summary   = "Performs full merge operation with left-on and right-on";

    let arguments = (ins LLVMPointerType:$srctable1, LLVMPointerType:$srctable2, StrAttr:$left_on_col, StrAttr:$righton_col);
    let results   = (outs LLVMPointerType:$dfref);

    let assemblyFormat = [{
    `(` $srctable1 `:` type($srctable1) `,` $srctable2 `:` type($srctable2) `,` $left_on_col `,` $righton_col `)` attr-dict type($dfref)
    }];

    let builders = [
        OpBuilder<(ins "std::string":$tablename)>
    ];
}


//===----------------------------------------------------------------------===//
// FilterDateGTDFOp
//===----------------------------------------------------------------------===//
def FilterDateGTDFOp : Pandas_Op<"filter_date_gt"> {
    let summary   = "Performs filter operation on dataframe with date predicate (>)";

    let arguments = (ins LLVMPointerType:$srctable, StrAttr:$columnname, StrAttr:$date, StrAttr:$newtable);
    let results   = (outs LLVMPointerType:$dfref);

    let assemblyFormat = [{
    `(` $srctable `:` type($srctable) `,` $columnname `,` $date `,` $newtable `)` attr-dict type($dfref)
    }];

    let builders = [
        OpBuilder<(ins "std::string":$tablename)>
    ];
}

//===----------------------------------------------------------------------===//
// FilterDateGEQDFOp
//===----------------------------------------------------------------------===//
def FilterDateGEQDFOp : Pandas_Op<"filter_date_geq"> {
    let summary   = "Performs filter operation on dataframe with date predicate (>=)";

    let arguments = (ins LLVMPointerType:$srctable, StrAttr:$columnname, StrAttr:$date);
    let results   = (outs LLVMPointerType:$dfref);

    let assemblyFormat = [{
    `(` $srctable `:` type($srctable) `,` $columnname `,`  $date `)` attr-dict type($dfref)
    }];

    let builders = [
        OpBuilder<(ins "std::string":$tablename)>
    ];
}

//===----------------------------------------------------------------------===//
// FilterDateLEQDFOp
//===----------------------------------------------------------------------===//
def FilterDateLEQDFOp : Pandas_Op<"filter_date_leq"> {
    let summary   = "Performs filter operation on dataframe with date predicate (<=)";

    let arguments = (ins LLVMPointerType:$srctable, StrAttr:$columnname, StrAttr:$date);
    let results   = (outs LLVMPointerType:$dfref);

    let assemblyFormat = [{
    `(` $srctable `:` type($srctable) `,` $columnname `,`  $date  `)` attr-dict type($dfref)
    }];

    let builders = [
        OpBuilder<(ins "std::string":$tablename)>
    ];
}

//===----------------------------------------------------------------------===//
// FilterDateLTDFOp
//===----------------------------------------------------------------------===//
def FilterDateLTDFOp : Pandas_Op<"filter_date_lt"> {
    let summary   = "Performs filter operation on dataframe with date predicate (<)";

    let arguments = (ins LLVMPointerType:$srctable, StrAttr:$columnname, StrAttr:$date);
    let results   = (outs LLVMPointerType:$dfref);

    let assemblyFormat = [{
    `(` $srctable `:` type($srctable) `,` $columnname `,`  $date  `)` attr-dict type($dfref)
    }];

    let builders = [
        OpBuilder<(ins "std::string":$tablename)>
    ];
}

//===----------------------------------------------------------------------===//
// FilterDateEQDFOp
//===----------------------------------------------------------------------===//
def FilterDateEQDFOp : Pandas_Op<"filter_date_eq"> {
    let summary   = "Performs filter operation on dataframe with date predicate (=)";

    let arguments = (ins LLVMPointerType:$srctable, StrAttr:$columnname, StrAttr:$date);
    let results   = (outs LLVMPointerType:$dfref);

    let assemblyFormat = [{
    `(` $srctable `:` type($srctable) `,` $columnname `,`  $date  `)` attr-dict type($dfref)
    }];

    let builders = [
        OpBuilder<(ins "std::string":$tablename)>
    ];
}


//===----------------------------------------------------------------------===//
// FilterNumGTDFOp
//===----------------------------------------------------------------------===//
def FilterNumGTDFOp : Pandas_Op<"filter_num_gt"> {
    let summary   = "Performs filter operation on dataframe with numeric field predicate (>)";

    let arguments = (ins LLVMPointerType:$srctable, StrAttr:$columnname, StrAttr:$numval);
    let results   = (outs LLVMPointerType:$dfref);

    let assemblyFormat = [{
    `(` $srctable `:` type($srctable) `,` $columnname `,`  $numval  `)` attr-dict type($dfref)
    }];

    let builders = [
        OpBuilder<(ins "std::string":$tablename)>
    ];
}
//===----------------------------------------------------------------------===//
// FilterNumGEQDFOp
//===----------------------------------------------------------------------===//
def FilterNumGEQDFOp : Pandas_Op<"filter_num_geq"> {
    let summary   = "Performs filter operation on dataframe with numeric field predicate (>=)";

    let arguments = (ins LLVMPointerType:$srctable, StrAttr:$columnname, StrAttr:$numval);
    let results   = (outs LLVMPointerType:$dfref);

    let assemblyFormat = [{
    `(` $srctable `:` type($srctable)`,` $columnname `,`  $numval `)` attr-dict type($dfref)
    }];

    let builders = [
        OpBuilder<(ins "std::string":$tablename)>
    ];
}
//===----------------------------------------------------------------------===//
// FilterNumLTDFOp
//===----------------------------------------------------------------------===//
def FilterNumLTDFOp : Pandas_Op<"filter_num_lt"> {
    let summary   = "Performs filter operation on dataframe with numeric field predicate (<)";

    let arguments = (ins LLVMPointerType:$srctable, StrAttr:$columnname, StrAttr:$numval);
    let results   = (outs LLVMPointerType:$dfref);

    let assemblyFormat = [{
    `(` $srctable `:` type($srctable)`,` $columnname `,`  $numval `)` attr-dict type($dfref)
    }];

    let builders = [
        OpBuilder<(ins "std::string":$tablename)>
    ];
}
//===----------------------------------------------------------------------===//
// FilterNumLEQDFOp
//===----------------------------------------------------------------------===//
def FilterNumLEQDFOp : Pandas_Op<"filter_num_leq"> {
    let summary   = "Performs filter operation on dataframe with numeric field predicate (<=)";

    let arguments = (ins LLVMPointerType:$srctable, StrAttr:$columnname, StrAttr:$numval);
    let results   = (outs LLVMPointerType:$dfref);

    let assemblyFormat = [{
    `(` $srctable `:` type($srctable)`,` $columnname `,`  $numval `)` attr-dict type($dfref)
    }];

    let builders = [
        OpBuilder<(ins "std::string":$tablename)>
    ];
}

//===----------------------------------------------------------------------===//
// FilterNumEQDFOp
//===----------------------------------------------------------------------===//
def FilterNumEQDFOp : Pandas_Op<"filter_num_eq"> {
    let summary   = "Performs filter operation on dataframe with numeric field predicate (=)";

    let arguments = (ins LLVMPointerType:$srctable, StrAttr:$columnname, StrAttr:$numval);
    let results   = (outs LLVMPointerType:$dfref);

    let assemblyFormat = [{
    `(` $srctable `:` type($srctable)`,` $columnname `,`  $numval `)` attr-dict type($dfref)
    }];

    let builders = [
        OpBuilder<(ins "std::string":$tablename)>
    ];
}

//===----------------------------------------------------------------------===//
// GroupBySumDFOp
//===----------------------------------------------------------------------===//
def GroupBySumDFOp : Pandas_Op<"groupby_sum"> {
    let summary   = "Performs GroupBy operation and aggregation/Sum over the same";

    let arguments = (ins LLVMPointerType:$srctable, StrAttr:$groupbycolname, StrAttr:$rescolname);
    let results   = (outs LLVMPointerType:$dfref);

    let assemblyFormat = [{
    `(` $srctable `:` type($srctable) `,` $groupbycolname `,`  $rescolname `)` attr-dict type($dfref)
    }];

    let builders = [
        OpBuilder<(ins "std::string":$tablename)>
    ];
}

//===----------------------------------------------------------------------===//
// GroupByAvgDFOp
//===----------------------------------------------------------------------===//
def GroupByAvgDFOp : Pandas_Op<"groupby_avg"> {
    let summary   = "Performs GroupBy operation and aggregation/avg over the same";

    let arguments = (ins LLVMPointerType:$srctable, StrAttr:$groupbycolname, StrAttr:$rescolname);
    let results   = (outs LLVMPointerType:$dfref);

    let assemblyFormat = [{
    `(` $srctable `:` type($srctable)  `,` $groupbycolname `,`  $rescolname `)` attr-dict type($dfref)
    }];

    let builders = [
        OpBuilder<(ins "std::string":$tablename)>
    ];
}



#endif // PANDAS_OPS
