#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "csi"

namespace {

const char *const CsiUnitCtorName = "csi.unit_ctor";
const char *const CsiRtUnitInitName = "__csirt_unit_init";
const int CsiRtUnitCtorPriority = 65535;

/// The Comprehensive Static Instrumentation pass.
/// Inserts calls to user-defined hooks at predefined points in the IR.
struct ComprehensiveStaticInstrumentation : public ModulePass {
  static char ID;

  ComprehensiveStaticInstrumentation() : ModulePass(ID) {}
  const char *getPassName() const override;
  bool runOnModule(Module &M) override;

private:
  void insertCsiUnitCtor(Module &M);
};
} // End anonymous namespace

char ComprehensiveStaticInstrumentation::ID = 0;

INITIALIZE_PASS(ComprehensiveStaticInstrumentation, "csi",
                "ComprehensiveStaticInstrumentation: inserts calls to "
                "user-defined hooks in the IR.",
                false, false)

const char *ComprehensiveStaticInstrumentation::getPassName() const {
  return "ComprehensiveStaticInstrumentation";
}

ModulePass *llvm::createComprehensiveStaticInstrumentationPass() {
  return new ComprehensiveStaticInstrumentation();
}

/// Add the CSI unit constructor, which calls __csirt_unit_init.
void ComprehensiveStaticInstrumentation::insertCsiUnitCtor(Module &M) {
  LLVMContext &C = M.getContext();
  Function *Ctor =
      Function::Create(FunctionType::get(Type::getVoidTy(C), false),
                       GlobalValue::InternalLinkage, CsiUnitCtorName, &M);
  BasicBlock *CtorBB = BasicBlock::Create(C, "", Ctor);
  IRBuilder<> IRB(ReturnInst::Create(C, CtorBB));

  // Lookup __csirt_unit_init
  SmallVector<Type *, 4> CsiRtUnitInitArgTypes({IRB.getInt8PtrTy()});
  FunctionType *CsiRtUnitInitFunctionTy =
      FunctionType::get(IRB.getVoidTy(), CsiRtUnitInitArgTypes, false);
  Function *CsiRtUnitInit = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction(CsiRtUnitInitName, CsiRtUnitInitFunctionTy));

  // Insert call in constructor to __csirt_unit_init
  IRB.CreateCall(CsiRtUnitInit, {IRB.CreateGlobalStringPtr(M.getName())});

  // Add the constructor to the global list
  appendToGlobalCtors(M, Ctor, CsiRtUnitCtorPriority);
}

bool ComprehensiveStaticInstrumentation::runOnModule(Module &M) {
  insertCsiUnitCtor(M);
  return true;
}
