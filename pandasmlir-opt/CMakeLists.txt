get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LLVM_LINK_COMPONENTS
        Core
        Support
        nativecodegen
        OrcJIT
  )

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRAnalysis
        MLIRBuiltinToLLVMIRTranslation
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRTargetLLVMIRExport
        MLIRTransforms
        MLIRFuncAllExtensions
        )
add_llvm_executable(pandasmlir-opt pandasmlir-opt.cpp)

llvm_update_compile_flags(pandasmlir-opt)
target_link_libraries(pandasmlir-opt PRIVATE ${LIBS} MLIRPandas)
add_dependencies(pandasmlir-opt DataLibC DataLibCXX TestsC TestsCXX)

mlir_check_all_link_libraries(pandasmlir-opt)
