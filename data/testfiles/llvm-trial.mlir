module {
    llvm.mlir.global constant @df_name("my_dataframe") : !llvm.array<12 x i8>
    llvm.mlir.global constant @csvpath_customer("/home/vaisakhps/developer/Compiler/E0_255-CD_MLIR-Project/tpc-benchmarks/tables_scale_1/customer.csv") : !llvm.array<100 x i8>
    llvm.mlir.global constant @csvpath_nation("/home/vaisakhps/developer/Compiler/E0_255-CD_MLIR-Project/tpc-benchmarks/tables_scale_1/nation.csv") : !llvm.array<98 x i8>
    llvm.mlir.global constant @csvpath_lineitem("/home/vaisakhps/developer/Compiler/E0_255-CD_MLIR-Project/tpc-benchmarks/tables_scale_1/lineitem.csv") : !llvm.array<100 x i8>

    llvm.func @DataFrame_new(%arg0: !llvm.ptr) -> (!llvm.ptr)
    llvm.func @DataFrame_free(%arg0: !llvm.ptr) -> ()
    llvm.func @DataFrame_read_csv(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> (i32)

    func.func @main() {
        %df_name_ptr = llvm.mlir.addressof @df_name : !llvm.ptr
        %csvpath_ptr = llvm.mlir.addressof @csvpath_lineitem : !llvm.ptr

        %df = llvm.call @DataFrame_new(%df_name_ptr) : (!llvm.ptr) -> (!llvm.ptr)
        %result = llvm.call @DataFrame_read_csv(%df, %csvpath_ptr) : (!llvm.ptr, !llvm.ptr) -> i32

        llvm.call @DataFrame_free(%df) : (!llvm.ptr) -> ()
        llvm.return
    }
}