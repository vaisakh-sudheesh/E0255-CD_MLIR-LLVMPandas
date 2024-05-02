module {

    func.func @main() {
        %0 = pandas.read_csv("lineitem_df", "/home/vaisakhps/developer/Compiler/E0_255-CD_MLIR-Project/tpc-benchmarks/tables_scale_1/lineitem.csv") : !llvm.ptr
        llvm.return
    }
}