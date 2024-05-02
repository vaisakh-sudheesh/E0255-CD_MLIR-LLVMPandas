#include "mlir/Dialect/Func/Extensions/AllExtensions.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"


#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "Pandas/PandasPasses.h"

namespace cl = llvm::cl;


static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { Toy, MLIR };
} // namespace

namespace {
    enum Action {
        None,
        DumpAST,
        DumpMLIR,
        DumpMLIRLLVM,
        DumpLLVMIR,
        RunJIT
    };
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(
        clEnumValN(RunJIT, "jit",
                   "JIT the code and run it by invoking the main function")));


int loadMLIR(mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {

    // Otherwise, the input is '.mlir'.
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    // Parse the input mlir.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    }
    return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context,
                       mlir::OwningOpRef<mlir::ModuleOp> &module) {
    if (int error = loadMLIR(context, module))
        return error;

    mlir::PassManager pm(module.get()->getName());
    // Apply any generic pass manager command line options and run the pipeline.
    if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
        return 4;

    // Check to see what granularity of MLIR we are compiling to.
    bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;


    if (isLoweringToLLVM) {
        llvm::errs()<<"Lowering to LLVM"<<"\n";
        // Finish lowering the toy IR to the LLVM dialect.
        pm.addPass(mlir::pandas::createLowerToLLVMPass());
        
        // This is necessary to have line tables emitted and basic
        // debugger working. In the future we will add proper debug information
        // emission directly from our frontend.
        pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(
            mlir::LLVM::createDIScopeForLLVMFuncOpPass());
    }

    if (mlir::failed(pm.run(*module)))
        return 4;
    return 0;
}


int dumpLLVMIR(mlir::ModuleOp module) {
    // Register the translation to LLVM IR with the MLIR context.
    mlir::registerBuiltinDialectTranslation(*module->getContext());
    mlir::registerLLVMDialectTranslation(*module->getContext());

    // Convert the module to LLVM IR in a new LLVM IR context.
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return -1;
    }

    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Create target machine and configure the LLVM Module
    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrError) {
        llvm::errs() << "Could not create JITTargetMachineBuilder\n";
        return -1;
    }

    auto tmOrError = tmBuilderOrError->createTargetMachine();
    if (!tmOrError) {
        llvm::errs() << "Could not create TargetMachine\n";
        return -1;
    }
    mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                            tmOrError.get().get());

    llvm::errs() << *llvmModule << "\n";
    return 0;
}


#define LIBRARY_SEARCH_PREFIX "/home/vaisakhps/developer/Compiler/E0_255-CD_MLIR-Project/opt/lib"
#define MLIR_LIBRARY(x) LIBRARY_SEARCH_PREFIX "/" x

//#define PROJECT_LIBRARY_SEARCH_PREFIX "/home/vaisakhps/developer/Compiler/E0_255-CD_MLIR-Project/pandas-mlir/build/lib"
#define PROJECT_LIBRARY_SEARCH_PREFIX "/home/vaisakhps/developer/CDProject-v2/build/lib"
#define PROJECT_LIBRARY(x) PROJECT_LIBRARY_SEARCH_PREFIX "/" x

int runJit(mlir::ModuleOp module) {
    mlir::registerAllPasses();

    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Register the translation from MLIR to LLVM IR, which must happen before we
    // can JIT-compile.
    mlir::registerBuiltinDialectTranslation(*module->getContext());
    mlir::registerLLVMDialectTranslation(*module->getContext());

    // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
    // the module.
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.sharedLibPaths = {
            PROJECT_LIBRARY("DataLib/libDataLibC.so"),
        };
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get(); 

    // Invoke the JIT-compiled function.
    auto invocationResult = engine->invokePacked("main");
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
        return -1;
    }
    return 0;
}


int main(int argc, char **argv) {
    // Register any command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();

    cl::ParseCommandLineOptions(argc, argv, "Pandas/MLIR compiler\n");

    // if (emitAction == Action::DumpAST)
    //     return dumpAST();

    // If we aren't dumping the AST, then we are compiling with/to MLIR.
    mlir::DialectRegistry registry;
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::pandas::PandasDialect>();
    // registerAllDialects(registry);
    mlir::func::registerAllExtensions(registry);

    mlir::MLIRContext context(registry);

    // Load our Dialect in this MLIR Context.

    mlir::OwningOpRef<mlir::ModuleOp> module;
    if (int error = loadAndProcessMLIR(context, module))
        return error;

    // If we aren't exporting to non-mlir, then we are done.
    bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
    if (isOutputingMLIR) {
        module->dump();
        return 0;
    }

    // Check to see if we are compiling to LLVM IR.
    if (emitAction == Action::DumpLLVMIR)
        return dumpLLVMIR(*module);

    // Otherwise, we must be running the jit.
    if (emitAction == Action::RunJIT)
        return runJit(*module);


    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    return -1;
}
