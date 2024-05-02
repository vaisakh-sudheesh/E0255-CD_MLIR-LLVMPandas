add_mlir_dialect(PandasOps pandas)
add_mlir_doc(PandasDialect PandasDialect Pandas/ -gen-dialect-doc)
add_mlir_doc(PandasOps PandasOps Pandas/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS PandasPasses.td)
mlir_tablegen(PandasPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRPandasPassesIncGen)
