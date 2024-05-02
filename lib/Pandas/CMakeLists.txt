add_mlir_dialect_library(MLIRPandas
        PandasTypes.cpp
        PandasDialect.cpp
        PandasOps.cpp
        PandasPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Pandas

        DEPENDS
        MLIRPandasOpsIncGen
        MLIRPandasPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
