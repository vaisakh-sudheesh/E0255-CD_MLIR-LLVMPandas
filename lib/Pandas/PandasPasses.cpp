#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Pandas/PandasPasses.h"
#include "Pandas/PandasDialect.h"
#include "Pandas/PandasOps.h"
#include "Pandas/PandasTypes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "llvm/ADT/Sequence.h"


using namespace mlir;
using namespace mlir::pandas;

//===----------------------------------------------------------------------===//
// ToyToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct PandasToLLVMLoweringPass
    : public PassWrapper<PandasToLLVMLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PandasToLLVMLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect,func::FuncDialect>();
  }
  void runOnOperation() final;
};
} // namespace

//===----------------------------------------------------------------------===//
// PandasToLLVMLoweringPass - ReadCSVOp
//===----------------------------------------------------------------------===//

namespace {
struct  ReadCSVOpLowering : public OneToNOpConversionPattern<mlir::pandas::ReadCSVOp> {
  using OneToNOpConversionPattern<mlir::pandas::ReadCSVOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pandas::ReadCSVOp op,  OpAdaptor adaptor,
                                OneToNPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    const OneToNTypeMapping &resultMapping = adaptor.getResultMapping();

    // auto readcsvRef = getOrInsertReadCSV(rewriter, parentModule);
    auto readcsvOp = cast<mlir::pandas::ReadCSVOp>(op);

    Value dfNameCst = getOrCreateGlobalString(loc, rewriter, 
                            generateUniqueName("tablename"), 
                            StringRef(readcsvOp.getTablenameAttr()), 
                            parentModule);
    auto szdfNameCst = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getIntegerAttr(rewriter.getI32Type(), readcsvOp.getTablenameAttr().size()));
    Value csvFileCst = getOrCreateGlobalString(loc, rewriter, 
                            generateUniqueName("csvfile"),
                            StringRef(readcsvOp.getFilename()), 
                            parentModule);
    auto szcsvFileCst = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getIntegerAttr(rewriter.getI32Type(), readcsvOp.getFilename().size()));

    auto readCsvRef = getOrInsert_DataFrame_read_csv(rewriter, parentModule);


// llvm::errs() << "csvFileCst "<< readcsvOp.getFilename().size()   <<":"<< readcsvOp.getFilename()<< "\n";
// llvm::errs() << "dfNameCst "<< readcsvOp.getTablenameAttr().size() <<":"<< readcsvOp.getTablename()<< "\n";
    
    // auto dfref = rewriter.create<func::CallOp>(loc, readCsvRef, rewriter.getI64Type(),dfNameCst);
    Value dfref = rewriter.create<LLVM::CallOp>(loc,
                                LLVM::LLVMPointerType::get(parentModule.getContext()),
                                readCsvRef, 
                                ArrayRef<Value>({dfNameCst, szdfNameCst, csvFileCst, szcsvFileCst})).getResult();

    // Notify the rewriter that this operation has been removed.
    rewriter.replaceOp(op, dfref, resultMapping);
    return success();
  }

private:
  // Function to generate a unique name based on a prefix and a unique identifier
  std::string generateUniqueName(const std::string &prefix) const {
    static std::atomic<int> counter(0);
    return prefix + "_" + std::to_string(counter.fetch_add(1));
  }

  static FlatSymbolRefAttr getOrInsert_DataFrame_free(PatternRewriter &rewriter, ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("DataFrame_free"))
      return SymbolRefAttr::get(context, "DataFrame_free");

    // Create a function declaration for DataFrame_free, the signature is:
    // void DataFrame_free(void* df);
    //   * `void (void*)`
    auto llvmVPtrTy = LLVM::LLVMPointerType::get(context);
    auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmVoidTy, llvmVPtrTy, /*isVarArg=*/false);

    // Insert the DataFrame_free function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "DataFrame_free", llvmFnType);
    return SymbolRefAttr::get(context, "DataFrame_free");
  }

  static FlatSymbolRefAttr getOrInsert_DataFrame_read_csv(PatternRewriter &rewriter, ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("DataFrame_read_csv"))
      return SymbolRefAttr::get(context, "DataFrame_read_csv");

    // Create a function declaration for read_csv, the signature is:
    // int DataFrame_read_csv(void* df, const char* csvFilePath);
    //   * `void* (i8*, i8*)`
    auto llvmVPtrTy = LLVM::LLVMPointerType::get(context);
    auto llvmI32Ty1 = IntegerType::get(context, 32);
    auto llvmI32Ty2 = IntegerType::get(context, 32);
    auto llvmI8PtrTy1 = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto llvmI8PtrTy2 = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    std::vector<Type> args;
    args.push_back(llvmI8PtrTy1);
    args.push_back(llvmI32Ty1);
    args.push_back(llvmI8PtrTy2);
    args.push_back(llvmI32Ty2);
    ArrayRef<Type> args_ar = ArrayRef(args);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmVPtrTy, args_ar, /*isVarArg=*/false);

    // Insert the DataFrame_read_csv function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "DataFrame_read_csv", llvmFnType);
    return SymbolRefAttr::get(context, "DataFrame_read_csv");
  }


  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
          IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value),
                                              /*alignment=*/0);
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                  builder.getIndexAttr(0));
    return builder.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
        globalPtr, ArrayRef<Value>({cst0, cst0}));
  }
};

//===----------------------------------------------------------------------===//
// PandasToLLVMLoweringPass - PrintDFOp
//===----------------------------------------------------------------------===//
struct PrintDFOpOpLowering : public OneToNOpConversionPattern<mlir::pandas::PrintDFOp>  {

  using OneToNOpConversionPattern<mlir::pandas::PrintDFOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pandas::PrintDFOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const final {
    const OneToNTypeMapping &resultMapping = adaptor.getResultMapping();
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    auto readCsvRef = getOrInsert_DataFrame_printrows(rewriter, parentModule);
    // if (!adaptor.getOperandMapping().hasNonIdentityConversion() &&
    //     !resultMapping.hasNonIdentityConversion())
    //   return failure();
    
    auto llvmPtrTyu1 = LLVM::LLVMPointerType::get(parentModule.getContext());
    rewriter.create<LLVM::CallOp>(loc,
                llvmPtrTyu1,
                readCsvRef, 
                ArrayRef<Value>({op->getOperand(0), op->getOperand(1)}));

    // Notify the rewriter that this operation has been removed.
    // rewriter.eraseOp(op);
    rewriter.eraseOp(op);
    return success();
  }

private:

  static FlatSymbolRefAttr getOrInsert_DataFrame_printrows(PatternRewriter &rewriter, ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("DataFrame_printRows"))
      return SymbolRefAttr::get(context, "DataFrame_printRows");

    // Create a function declaration for read_csv, the signature is:
    // void  DataFrame_printRows(void* df, int n);
    //   * `void (void*, i32)`
    auto llvmVPtrTy = LLVM::LLVMPointerType::get(context);
    auto llvmI32Ty1 = IntegerType::get(context, 32);
    auto llvmVPtrTy2 = LLVM::LLVMPointerType::get(context);
  
    std::vector<Type> args;
    args.push_back(llvmVPtrTy);
    args.push_back(llvmI32Ty1);
    ArrayRef<Type> args_ar = ArrayRef(args);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmVPtrTy2, args_ar, /*isVarArg=*/false);

    // Insert the DataFrame_printRows function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "DataFrame_printRows", llvmFnType);
    return SymbolRefAttr::get(context, "DataFrame_printRows");
  }
};

//===----------------------------------------------------------------------===//
// PandasToLLVMLoweringPass - MergeOnDFOp
//===----------------------------------------------------------------------===//
struct MergeOnDFOpOpLowering : public OneToNOpConversionPattern<mlir::pandas::MergeOnDFOp>  {

  using OneToNOpConversionPattern<mlir::pandas::MergeOnDFOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pandas::MergeOnDFOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const final {
    const OneToNTypeMapping &resultMapping = adaptor.getResultMapping();
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto mergeOp = cast<mlir::pandas::MergeOnDFOp>(op);

    Value dfcolumn = getOrCreateGlobalString(loc, rewriter, 
                            generateUniqueName("colname"), 
                            StringRef(mergeOp.getColnameAttr()), 
                            parentModule);
    auto szdfcolumn = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getIntegerAttr(rewriter.getI32Type(), mergeOp.getColnameAttr().size()));

    auto mergeRef = getOrInsert_DataFrame_merge(rewriter, parentModule);    
    auto llvmPtrTyu1 = LLVM::LLVMPointerType::get(parentModule.getContext());
    Value dfref = rewriter.create<LLVM::CallOp>(loc,
                llvmPtrTyu1,
                mergeRef, 
                ArrayRef<Value>({op->getOperand(0), op->getOperand(1), dfcolumn, szdfcolumn})).getResult();

    rewriter.replaceOp(op, dfref, resultMapping);
    
    return success();
  }
private:
  // Function to generate a unique name based on a prefix and a unique identifier
  std::string generateUniqueName(const std::string &prefix) const {
    static std::atomic<int> counter(0);
    return prefix + "_" + std::to_string(counter.fetch_add(1));
  }
  
    static FlatSymbolRefAttr getOrInsert_DataFrame_merge(PatternRewriter &rewriter, ModuleOp module) {
      auto *context = module.getContext();
      if (module.lookupSymbol<LLVM::LLVMFuncOp>("DataFrame_merge"))
        return SymbolRefAttr::get(context, "DataFrame_merge");
  
      // Create a function declaration for read_csv, the signature is:
      // void* DataFrame_merge(void* df1, void* df2, const char* newTableName, int sznewTableName);
      //   * `void (void*, void*, i8*, i32)`
      auto llvmVPtrTy1 = LLVM::LLVMPointerType::get(context);
      
      auto llvmVPtrTy2 = LLVM::LLVMPointerType::get(context);
      auto llvmVPtrTy3 = LLVM::LLVMPointerType::get(context);
      auto llvmI8PtrTy1 = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
      auto llvmI32Ty2 = IntegerType::get(context, 32);
    
      std::vector<Type> args;
      args.push_back(llvmVPtrTy2);
      args.push_back(llvmVPtrTy3);
      args.push_back(llvmI8PtrTy1);
      args.push_back(llvmI32Ty2);
      ArrayRef<Type> args_ar = ArrayRef(args);
      auto llvmFnType = LLVM::LLVMFunctionType::get(llvmVPtrTy1, args_ar, /*isVarArg=*/false);
  
      // Insert the DataFrame_printRows function into the body of the parent module.
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "DataFrame_merge", llvmFnType);
      return SymbolRefAttr::get(context, "DataFrame_merge");
    }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
          IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value),
                                              /*alignment=*/0);
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                  builder.getIndexAttr(0));
    return builder.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
        globalPtr, ArrayRef<Value>({cst0, cst0}));
  }
  };

//===----------------------------------------------------------------------===//
// PandasToLLVMLoweringPass - FilterDateGTDFOp
//===----------------------------------------------------------------------===//
struct FilterDateGTDFOpOpLowering : public OneToNOpConversionPattern<mlir::pandas::FilterDateGTDFOp>  {

  using OneToNOpConversionPattern<mlir::pandas::FilterDateGTDFOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pandas::FilterDateGTDFOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const final {
    const OneToNTypeMapping &resultMapping = adaptor.getResultMapping();
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto filterOp = cast<mlir::pandas::FilterDateGTDFOp>(op);

    Value dfcolumn = getOrCreateGlobalString(loc, rewriter, 
                            generateUniqueName("filtergt_colname"), 
                            StringRef(filterOp.getColumnnameAttr()), 
                            parentModule);
    auto szdfcolumn = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getIntegerAttr(rewriter.getI32Type(), filterOp.getColumnnameAttr().size()));

    Value dfdate = getOrCreateGlobalString(loc, rewriter, 
                            generateUniqueName("filtergt_date"), 
                            StringRef(filterOp.getDateAttr()), 
                            parentModule);
    auto szdfdate = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getIntegerAttr(rewriter.getI32Type(), filterOp.getDateAttr().size()));

    Value dfnewtable = getOrCreateGlobalString(loc, rewriter, 
                            generateUniqueName("filtergt_newtable"), 
                            StringRef(filterOp.getNewtableAttr()), 
                            parentModule);
    auto szdfnewtable = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getIntegerAttr(rewriter.getI32Type(), filterOp.getNewtableAttr().size()));

    Value dfpredicate = getOrCreateGlobalString(loc, rewriter, 
                            "filtergt_predicate", 
                            StringRef(">"), 
                            parentModule);
    auto szdfpredicate = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getIntegerAttr(rewriter.getI32Type(), 1));


    auto filterRef = getOrInsert_DataFrame_filter(rewriter, parentModule);    
    auto llvmPtrTyu1 = LLVM::LLVMPointerType::get(parentModule.getContext());
    Value dfref = rewriter.create<LLVM::CallOp>(loc,
                llvmPtrTyu1,
                filterRef, 
                ArrayRef<Value>({op->getOperand(0), 
                            dfcolumn, szdfcolumn, 
                            dfpredicate, szdfpredicate,
                            dfdate, szdfdate,
                            dfnewtable, szdfnewtable })).getResult();

    rewriter.replaceOp(op, dfref, resultMapping);
    return success();
  }

private:
  // Function to generate a unique name based on a prefix and a unique identifier
  std::string generateUniqueName(const std::string &prefix) const {
    static std::atomic<int> counter(0);
    return prefix + "_" + std::to_string(counter.fetch_add(1));
  }
  
    static FlatSymbolRefAttr getOrInsert_DataFrame_filter(PatternRewriter &rewriter, ModuleOp module) {
      auto *context = module.getContext();
      if (module.lookupSymbol<LLVM::LLVMFuncOp>("DataFrame_filter"))
        return SymbolRefAttr::get(context, "DataFrame_filter");
  
      // Create a function declaration for read_csv, the signature is:
      // void* DataFrame_filter(void* df, const char* columnName_, int szcolumnName,
      //                  const char* predicate_, int szpredicate,
      //                  const char* value_, int szvalue,
      //                  const char* newTableName_, int sznewTableName
      //                  );
      //   * `void (void*, i8*, i32, i8*, i32,, i8*, i32,, i8*, i32,)`
      auto llvmVPtrTy1 = LLVM::LLVMPointerType::get(context);
      auto llvmVPtrTy2 = LLVM::LLVMPointerType::get(context);

      auto llvmI8PtrTy1 = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
      auto llvmI32Ty1 = IntegerType::get(context, 32);

      auto llvmI8PtrTy2 = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
      auto llvmI32Ty2 = IntegerType::get(context, 32);

      auto llvmI8PtrTy3 = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
      auto llvmI32Ty3 = IntegerType::get(context, 32);

      auto llvmI8PtrTy4 = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
      auto llvmI32Ty4 = IntegerType::get(context, 32);
      
    
      std::vector<Type> args;
      args.push_back(llvmVPtrTy1);
      args.push_back(llvmI8PtrTy1);
      args.push_back(llvmI32Ty1);
      args.push_back(llvmI8PtrTy2);
      args.push_back(llvmI32Ty2);
      args.push_back(llvmI8PtrTy3);
      args.push_back(llvmI32Ty3);
      args.push_back(llvmI8PtrTy4);
      args.push_back(llvmI32Ty4);
      ArrayRef<Type> args_ar = ArrayRef(args);
      auto llvmFnType = LLVM::LLVMFunctionType::get(llvmVPtrTy2, args_ar, /*isVarArg=*/false);
  
      // Insert the DataFrame_printRows function into the body of the parent module.
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "DataFrame_filter", llvmFnType);
      return SymbolRefAttr::get(context, "DataFrame_filter");
    }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder, StringRef name, StringRef value, ModuleOp module) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, name, builder.getStringAttr(value), /*alignment=*/0);
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(), builder.getIndexAttr(0));
    return builder.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)), globalPtr, ArrayRef<Value>({cst0, cst0}));
  }
};


} // namespace

void PandasToLLVMLoweringPass::runOnOperation() {

  // ConversionTarget target(getContext());
  LLVMConversionTarget target(getContext());
  target.addLegalDialect< BuiltinDialect,arith::ArithDialect,
                         func::FuncDialect, LLVM::LLVMDialect>();
  target.addIllegalDialect<pandas::PandasDialect>();
  target.addLegalOp<ModuleOp>();

  LLVMTypeConverter typeConverter(&getContext());

  RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);

  populateMathToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);

  populateVectorToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  patterns.add<
      ReadCSVOpLowering, 
      PrintDFOpOpLowering,
      MergeOnDFOpOpLowering,
      FilterDateGTDFOpOpLowering
    > (typeConverter, &getContext());

  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::pandas::createLowerToLLVMPass() {
  return std::make_unique<PandasToLLVMLoweringPass>();
}
