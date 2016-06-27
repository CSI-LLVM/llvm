#include "llvm/IR/Module.h"
#include "llvm/Transforms/Instrumentation.h"

using namespace llvm;

#define DEBUG_TYPE "csi"

namespace {
/// The Comprehensive Static Instrumentation pass.
/// Inserts calls to user-defined hooks at predefined points in the IR.
struct ComprehensiveStaticInstrumentation : public ModulePass {
  static char ID;

  ComprehensiveStaticInstrumentation() : ModulePass(ID) {}
  const char *getPassName() const override;
  bool runOnModule(Module &M) override;
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

bool ComprehensiveStaticInstrumentation::runOnModule(Module &M) {
  return false;
}
