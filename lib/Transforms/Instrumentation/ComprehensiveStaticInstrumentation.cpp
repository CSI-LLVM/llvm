#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

namespace {
const char *const CsiRtUnitInitName = "__csirt_unit_init";
const char *const CsiRtUnitCtorName = "csirt.unit_ctor";
const char *const CsiFunctionBaseIdName = "__csi_unit_func_base_id";
const char *const CsiFunctionExitBaseIdName = "__csi_unit_func_exit_base_id";
const char *const CsiBasicBlockBaseIdName = "__csi_unit_bb_base_id";
const char *const CsiCallsiteBaseIdName = "__csi_unit_callsite_base_id";
const char *const CsiLoadBaseIdName = "__csi_unit_load_base_id";
const char *const CsiStoreBaseIdName = "__csi_unit_store_base_id";
const char *const CsiUnitFedTableName = "__csi_unit_fed_table";
const char *const CsiFuncIdVariablePrefix = "__csi_func_id_";
const char *const CsiUnitFedTableArrayName = "__csi_unit_fed_tables";
const char *const CsiInitCallsiteToFunctionName = "__csi_init_callsite_to_function";

const int64_t CsiCallsiteUnknownTargetId = -1;
// See llvm/tools/clang/lib/CodeGen/CodeGenModule.h:
const int CsiUnitCtorPriority = 65535;

DILocation *getFirstDebugLoc(BasicBlock &BB) {
  for (Instruction &Inst : BB)
    if (DILocation *Loc = Inst.getDebugLoc())
      return Loc;

  return nullptr;
}

class FrontEndDataTable {
public:
  FrontEndDataTable() {}
  FrontEndDataTable(Module &M, StringRef BaseIdName) {
    LLVMContext &C = M.getContext();
    IntegerType *Int64Ty = IntegerType::get(C, 64);
    GlobalVariable *GV = new GlobalVariable(M, Int64Ty, false, GlobalValue::InternalLinkage, ConstantInt::get(Int64Ty, 0), BaseIdName);
    assert(GV);
    idSpace = IdSpace(GV);
  }

  uint64_t size() const { return entries.size(); }

  GlobalVariable *baseId() const {
    return idSpace.base();
  }

  uint64_t add(Function &F) {
    uint64_t Id = add(F.getSubprogram());
    valueToLocalIdMap[&F] = Id;
    return Id;
  }

  uint64_t add(BasicBlock &BB) {
    uint64_t Id = add(getFirstDebugLoc(BB));
    valueToLocalIdMap[&BB] = Id;
    return Id;
  }

  uint64_t add(Instruction &I) {
    uint64_t Id = add(I.getDebugLoc());
    valueToLocalIdMap[&I] = Id;
    return Id;
  }

  uint64_t getId(Value *V) {
    assert(valueToLocalIdMap.find(V) != valueToLocalIdMap.end() && "Value not in ID map.");
    return valueToLocalIdMap[V];
  }

  Value *localToGlobalId(uint64_t LocalId, IRBuilder<> IRB) const {
    return idSpace.localToGlobalId(LocalId, IRB);
  }

  static PointerType *getPointerType(LLVMContext &C) {
    return PointerType::get(getEntryStructType(C), 0);
  }

  Constant *insertIntoModule(Module &M) const {
    LLVMContext &C = M.getContext();
    StructType *FedType = getEntryStructType(C);
    IntegerType *Int32Ty = IntegerType::get(C, 32);

    Constant *Zero = ConstantInt::get(Int32Ty, 0);
    Value *GepArgs[] = {Zero, Zero};

    IRBuilder<> IRB(C);
    SmallVector<Constant *, 4> EntryConstants;

    for (EntryList::const_iterator it = entries.cbegin(), ite = entries.cend(); it != ite; ++it) {
      const Entry &E = it->second;
      Value *Line = ConstantInt::get(Int32Ty, E.Line);

      // TODO(ddoucet): It'd be nice to reuse the global variables since most
      // module names will be the same. Do the pointers have the same value as well
      // or do we actually have to hash the string?
      Constant *FileStrConstant = ConstantDataArray::getString(C, E.File);
      GlobalVariable *GV = new GlobalVariable(M, FileStrConstant->getType(),
                                              true, GlobalValue::PrivateLinkage,
                                              FileStrConstant, "", nullptr,
                                              GlobalVariable::NotThreadLocal, 0);
      GV->setUnnamedAddr(true);
      Constant *File = ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs);

      EntryConstants.push_back(ConstantStruct::get(FedType, Line, File, nullptr));
    }

    ArrayType *FedArrayType = ArrayType::get(getEntryStructType(C), EntryConstants.size());
    Constant *Table = ConstantArray::get(FedArrayType, EntryConstants);
    GlobalVariable *GV = new GlobalVariable(M, FedArrayType, false, GlobalValue::InternalLinkage, Table, CsiUnitFedTableName);
    return ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs);
  }

private:
  class IdSpace {
  public:
    IdSpace() : BaseId(nullptr), IdCounter(0) {}
    IdSpace(GlobalVariable *Base) : BaseId(Base), IdCounter(0) {}

    uint64_t getNextLocalId() {
      return IdCounter++;
    }

    Value *localToGlobalId(uint64_t Id, IRBuilder<> IRB) const {
      assert(BaseId);
      Value *Base = IRB.CreateLoad(BaseId);
      Value *Offset = IRB.getInt64(Id);
      return IRB.CreateAdd(Base, Offset);
    }

    GlobalVariable *base() const {
      assert(BaseId);
      return BaseId;
    }
  private:
    GlobalVariable *BaseId;
    uint64_t IdCounter;
  };

  struct Entry {
    int32_t Line;
    StringRef File;
  };

  typedef std::map<uint64_t, Entry> EntryList;

  EntryList entries;
  IdSpace idSpace;
  std::map<Value *, uint64_t> valueToLocalIdMap;

  // Create a struct type to match the "struct source_loc_t" defined in csirt.c
  static StructType *getEntryStructType(LLVMContext &C) {
    return StructType::get(IntegerType::get(C, 32),
                           PointerType::get(IntegerType::get(C, 8), 0),
                           nullptr);
  }

  uint64_t add(DILocation *Loc) {
    if (Loc) {
      return add((int32_t)Loc->getLine(), Loc->getFilename());
    } else {
      return add(-1, "");
    }
  }

  uint64_t add(DISubprogram *Subprog) {
    if (Subprog) {
      return add((int32_t)Subprog->getLine(), Subprog->getFilename());
    } else {
      return add(-1, "");
    }
  }

  uint64_t add(int32_t Line, StringRef File) {
    uint64_t Id = idSpace.getNextLocalId();
    assert(entries.find(Id) == entries.end() && "Id already exists in FED table.");
    entries[Id] = { Line, File };
    return Id;
  }
};

typedef struct {
  unsigned unused;
  bool unused2, unused3;
  bool read_before_write_in_bb;
} csi_acc_prop_t;

struct ComprehensiveStaticInstrumentation : public ModulePass {
  static char ID;

  ComprehensiveStaticInstrumentation() : ModulePass(ID) {}
  const char *getPassName() const override;
  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  void initializeLoadStoreHooks(Module &M);
  void initializeFuncHooks(Module &M);
  void initializeBasicBlockHooks(Module &M);
  void initializeCallsiteHooks(Module &M);
  void initializeFEDTables(Module &M);
  void generateInitCallsiteToFunction(Module &M);

  int getNumBytesAccessed(Value *Addr, const DataLayout &DL);
  void computeAttributesForMemoryAccesses(
      SmallVectorImpl<std::pair<BasicBlock::iterator, csi_acc_prop_t> > &Accesses,
      SmallVectorImpl<BasicBlock::iterator> &LocalAccesses);

  void addLoadStoreInstrumentation(BasicBlock::iterator Iter,
                                   Function *BeforeFn,
                                   Function *AfterFn,
                                   Value *CsiId,
                                   Type *AddrType,
                                   Value *Addr,
                                   int NumBytes,
                                   csi_acc_prop_t prop);

  void instrumentLoadOrStore(BasicBlock::iterator Iter, csi_acc_prop_t prop, const DataLayout &DL);
  void instrumentMemIntrinsic(BasicBlock::iterator I);
  void instrumentCallsite(BasicBlock::iterator I);
  void instrumentBasicBlock(BasicBlock &BB);
  void instrumentFunction(Function &F);

  bool shouldNotInstrumentFunction(Function &F);
  void initializeCsi(Module &M);
  void finalizeCsi(Module &M);

  FrontEndDataTable FunctionFED, FunctionExitFED, BasicBlockFED,
    CallsiteFED, LoadFED, StoreFED;

  Function *CsiBeforeCallsite, *CsiAfterCallsite;
  Function *CsiFuncEntry, *CsiFuncExit;
  Function *CsiBBEntry, *CsiBBExit;
  Function *CsiBeforeRead, *CsiAfterRead;
  Function *CsiBeforeWrite, *CsiAfterWrite;

  CallGraph *CG;
  Function *MemmoveFn, *MemcpyFn, *MemsetFn;
  Function *InitCallsiteToFunction;
  Type *IntptrTy;
  std::map<std::string, uint64_t> FuncOffsetMap;
}; //struct ComprehensiveStaticInstrumentation
} // anonymous namespace

char ComprehensiveStaticInstrumentation::ID = 0;

INITIALIZE_PASS(ComprehensiveStaticInstrumentation, "csi",
                "ComprehensiveStaticInstrumentation pass", false, false)

const char *ComprehensiveStaticInstrumentation::getPassName() const {
  return "ComprehensiveStaticInstrumentation";
}

ModulePass *llvm::createComprehensiveStaticInstrumentationPass() {
  return new ComprehensiveStaticInstrumentation();
}

void ComprehensiveStaticInstrumentation::initializeFuncHooks(Module &M) {
  IRBuilder<> IRB(M.getContext());
  CsiFuncEntry = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_func_entry", IRB.getVoidTy(), IRB.getInt64Ty(), nullptr));
  CsiFuncExit = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_func_exit", IRB.getVoidTy(), IRB.getInt64Ty(), IRB.getInt64Ty(), nullptr));
}

void ComprehensiveStaticInstrumentation::initializeBasicBlockHooks(Module &M) {
  IRBuilder<> IRB(M.getContext());
  SmallVector<Type *, 4> ArgTypes({IRB.getInt64Ty()});
  FunctionType *FnType = FunctionType::get(IRB.getVoidTy(), ArgTypes, false);
  CsiBBEntry = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_bb_entry", FnType));

  CsiBBExit = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_bb_exit", FnType));
}

void ComprehensiveStaticInstrumentation::initializeCallsiteHooks(Module &M) {
  IRBuilder<> IRB(M.getContext());
  SmallVector<Type *, 4> ArgTypes({IRB.getInt64Ty(), IRB.getInt64Ty()});
  FunctionType *FnType = FunctionType::get(IRB.getVoidTy(), ArgTypes, false);
  CsiBeforeCallsite = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_call", FnType));
  CsiAfterCallsite = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_call", FnType));
}

void ComprehensiveStaticInstrumentation::initializeLoadStoreHooks(Module &M) {

  IRBuilder<> IRB(M.getContext());
  Type *RetType = IRB.getVoidTy();
  Type *AddrType = IRB.getInt8PtrTy();
  Type *NumBytesType = IRB.getInt32Ty();

  CsiBeforeRead = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_load", RetType,
        IRB.getInt64Ty(), AddrType, NumBytesType, IRB.getInt64Ty(), nullptr));

  SmallString<32> AfterReadName("__csi_after_load");
  CsiAfterRead = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_load", RetType,
        IRB.getInt64Ty(), AddrType, NumBytesType, IRB.getInt64Ty(), nullptr));

  CsiBeforeWrite = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_store", RetType,
        IRB.getInt64Ty(), AddrType, NumBytesType, IRB.getInt64Ty(), nullptr));

  CsiAfterWrite = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_store", RetType,
        IRB.getInt64Ty(), AddrType, NumBytesType, IRB.getInt64Ty(), nullptr));

  MemmoveFn = checkCsiInterfaceFunction(
      M.getOrInsertFunction("memmove", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                            IRB.getInt8PtrTy(), IntptrTy, nullptr));
  MemcpyFn = checkCsiInterfaceFunction(
      M.getOrInsertFunction("memcpy", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                            IRB.getInt8PtrTy(), IntptrTy, nullptr));
  MemsetFn = checkCsiInterfaceFunction(
      M.getOrInsertFunction("memset", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                            IRB.getInt32Ty(), IntptrTy, nullptr));
}

int ComprehensiveStaticInstrumentation::getNumBytesAccessed(Value *Addr,
                                                const DataLayout &DL) {
  Type *OrigPtrTy = Addr->getType();
  Type *OrigTy = cast<PointerType>(OrigPtrTy)->getElementType();
  assert(OrigTy->isSized());
  uint32_t TypeSize = DL.getTypeStoreSizeInBits(OrigTy);
  if (TypeSize != 8  && TypeSize != 16 && TypeSize != 32 && TypeSize != 64 && TypeSize != 128) {
    return -1;
  }
  return TypeSize / 8;
}

void ComprehensiveStaticInstrumentation::addLoadStoreInstrumentation(BasicBlock::iterator Iter,
                                                         Function *BeforeFn,
                                                         Function *AfterFn,
                                                         Value *CsiId,
                                                         Type *AddrType,
                                                         Value *Addr,
                                                         int NumBytes,
                                                         csi_acc_prop_t prop) {
  IRBuilder<> IRB(&(*Iter));
  IRB.CreateCall(BeforeFn,
      // XXX: should I just use the pointer type with the right size?
      {CsiId,
       IRB.CreatePointerCast(Addr, AddrType),
       IRB.getInt32(NumBytes),
       IRB.getInt64(0)});  // TODO(ddoucet): fix this
       /* IRB.getInt32(prop.unused),
       IRB.getInt1(prop.unused2),
       IRB.getInt1(prop.unused3),
       IRB.getInt1(prop.read_before_write_in_bb)}); */

  // The iterator currently points between the inserted instruction and the
  // store instruction. We now want to insert an instruction after the store
  // instruction.
  Iter++;
  IRB.SetInsertPoint(&*Iter);

  IRB.CreateCall(AfterFn,
      {CsiId,
       IRB.CreatePointerCast(Addr, AddrType),
       IRB.getInt32(NumBytes),
       IRB.getInt64(0)});  // TODO(ddoucet): fix this
       /* IRB.getInt32(prop.unused),
       IRB.getInt1(prop.unused2),
       IRB.getInt1(prop.unused3),
       IRB.getInt1(prop.read_before_write_in_bb)}); */
}

void ComprehensiveStaticInstrumentation::instrumentLoadOrStore(BasicBlock::iterator Iter,
                                                   csi_acc_prop_t prop,
                                                   const DataLayout &DL) {
  Instruction *I = &*Iter;
  IRBuilder<> IRB(I);
  bool IsWrite = isa<StoreInst>(I);
  Value *Addr = IsWrite ?
      cast<StoreInst>(I)->getPointerOperand()
      : cast<LoadInst>(I)->getPointerOperand();

  int NumBytes = getNumBytesAccessed(Addr, DL);
  Type *AddrType = IRB.getInt8PtrTy();

  if (NumBytes == -1) return; // size that we don't recognize

  if(IsWrite) {
    uint64_t LocalId = StoreFED.add(*I);
    Value *CsiId = StoreFED.localToGlobalId(LocalId, IRB);
    addLoadStoreInstrumentation(
        Iter, CsiBeforeWrite, CsiAfterWrite, CsiId, AddrType, Addr, NumBytes, prop);
  } else { // is read
    uint64_t LocalId = LoadFED.add(*I);
    Value *CsiId = LoadFED.localToGlobalId(LocalId, IRB);
    addLoadStoreInstrumentation(
        Iter, CsiBeforeRead, CsiAfterRead, CsiId, AddrType, Addr, NumBytes, prop);
  }
}

// If a memset intrinsic gets inlined by the code gen, we will miss races on it.
// So, we either need to ensure the intrinsic is not inlined, or instrument it.
// We do not instrument memset/memmove/memcpy intrinsics (too complicated),
// instead we simply replace them with regular function calls, which are then
// intercepted by the run-time.
// Since our pass runs after everyone else, the calls should not be
// replaced back with intrinsics. If that becomes wrong at some point,
// we will need to call e.g. __csi_memset to avoid the intrinsics.
void ComprehensiveStaticInstrumentation::instrumentMemIntrinsic(BasicBlock::iterator Iter) {
  Instruction *I = &(*Iter);
  IRBuilder<> IRB(I);
  if (MemSetInst *M = dyn_cast<MemSetInst>(I)) {
    IRB.CreateCall(
        MemsetFn,
        {IRB.CreatePointerCast(M->getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(M->getArgOperand(1), IRB.getInt32Ty(), false),
         IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false)});
    I->eraseFromParent();
  } else if (MemTransferInst *M = dyn_cast<MemTransferInst>(I)) {
    IRB.CreateCall(
        isa<MemCpyInst>(M) ? MemcpyFn : MemmoveFn,
        {IRB.CreatePointerCast(M->getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreatePointerCast(M->getArgOperand(1), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false)});
    I->eraseFromParent();
  }
}

void ComprehensiveStaticInstrumentation::instrumentBasicBlock(BasicBlock &BB) {
  IRBuilder<> IRB(&*BB.getFirstInsertionPt());
  uint64_t LocalId = BasicBlockFED.add(BB);
  Value *CsiId = BasicBlockFED.localToGlobalId(LocalId, IRB);

  IRB.CreateCall(CsiBBEntry, {CsiId});

  TerminatorInst *TI = BB.getTerminator();
  IRB.SetInsertPoint(TI);
  IRB.CreateCall(CsiBBExit, {CsiId});
}

void ComprehensiveStaticInstrumentation::instrumentCallsite(BasicBlock::iterator Iter) {
  IRBuilder<> IRB(&*Iter);
  CallSite CS(&*Iter);
  Instruction *Inst = CS.getInstruction();
  Module *M = Inst->getParent()->getParent()->getParent();
  Function *Called = CS.getCalledFunction();

  if (Called && Called->getName().startswith("llvm.dbg")) {
      return;
  }

  uint64_t LocalId = CallsiteFED.add(*Inst);
  Value *CallsiteId = CallsiteFED.localToGlobalId(LocalId, IRB);

  std::string GVName = CsiFuncIdVariablePrefix + Called->getName().str();
  GlobalVariable *FuncIdGV = dyn_cast<GlobalVariable>(M->getOrInsertGlobal(GVName, IRB.getInt64Ty()));
  assert(FuncIdGV);
  FuncIdGV->setConstant(false);
  FuncIdGV->setLinkage(GlobalValue::WeakAnyLinkage);
  FuncIdGV->setInitializer(IRB.getInt64(CsiCallsiteUnknownTargetId));

  Value *FuncId = IRB.CreateLoad(FuncIdGV);
  IRB.CreateCall(CsiBeforeCallsite, {CallsiteId, FuncId});

  Iter++;
  IRB.SetInsertPoint(&*Iter);
  IRB.CreateCall(CsiAfterCallsite, {CallsiteId, FuncId});
}

void ComprehensiveStaticInstrumentation::initializeFEDTables(Module &M) {
  FunctionFED = FrontEndDataTable(M, CsiFunctionBaseIdName);
  FunctionExitFED = FrontEndDataTable(M, CsiFunctionExitBaseIdName);
  BasicBlockFED = FrontEndDataTable(M, CsiBasicBlockBaseIdName);
  CallsiteFED = FrontEndDataTable(M, CsiCallsiteBaseIdName);
  LoadFED = FrontEndDataTable(M, CsiLoadBaseIdName);
  StoreFED = FrontEndDataTable(M, CsiStoreBaseIdName);
}

void ComprehensiveStaticInstrumentation::generateInitCallsiteToFunction(Module &M) {
  LLVMContext &C = M.getContext();
  BasicBlock *EntryBB = BasicBlock::Create(C, "", InitCallsiteToFunction);
  IRBuilder<> IRB(ReturnInst::Create(C, EntryBB));

  GlobalVariable *Base = FunctionFED.baseId();
  LoadInst *LI = IRB.CreateLoad(Base);
  for (const auto &it : FuncOffsetMap) {
    std::string GVName = CsiFuncIdVariablePrefix + it.first;
    GlobalVariable *GV = nullptr;
    if ((GV = M.getGlobalVariable(GVName)) == nullptr) {
      GV = new GlobalVariable(M, IRB.getInt64Ty(), false, GlobalValue::WeakAnyLinkage, IRB.getInt64(CsiCallsiteUnknownTargetId), GVName);
    }
    assert(GV);
    IRB.CreateStore(IRB.CreateAdd(LI, IRB.getInt64(it.second)), GV);
  }
}

void ComprehensiveStaticInstrumentation::initializeCsi(Module &M) {
  initializeFEDTables(M);
  initializeFuncHooks(M);
  initializeLoadStoreHooks(M);
  initializeBasicBlockHooks(M);
  initializeCallsiteHooks(M);

  FunctionType *FnType = FunctionType::get(Type::getVoidTy(M.getContext()), {}, false);
  InitCallsiteToFunction = checkCsiInterfaceFunction(M.getOrInsertFunction(CsiInitCallsiteToFunctionName, FnType));
  assert(InitCallsiteToFunction);
  InitCallsiteToFunction->setLinkage(GlobalValue::InternalLinkage);

  CG = &getAnalysis<CallGraphWrapperPass>().getCallGraph();
  IntptrTy = M.getDataLayout().getIntPtrType(M.getContext());
}

// Create a struct type to match the unit_fed_entry_t type in csirt.c.
StructType *getUnitFedTableType(LLVMContext &C, PointerType *EntryPointerType) {
  return StructType::get(IntegerType::get(C, 64),
                         PointerType::get(IntegerType::get(C, 64), 0),
                         EntryPointerType,
                         nullptr);
}

Constant *fedTableToUnitFedTable(Module &M,
                                 StructType *UnitFedTableType,
                                 FrontEndDataTable &FedTable) {
  Constant *NumEntries = ConstantInt::get(IntegerType::get(M.getContext(), 64), FedTable.size());
  Constant *InsertedTable = FedTable.insertIntoModule(M);
  return ConstantStruct::get(UnitFedTableType, NumEntries, FedTable.baseId(), InsertedTable, nullptr);
}

void ComprehensiveStaticInstrumentation::finalizeCsi(Module &M) {
  LLVMContext &C = M.getContext();

  // Add CSI global constructor, which calls unit init.
  Function *Ctor = Function::Create(
      FunctionType::get(Type::getVoidTy(C), false),
      GlobalValue::InternalLinkage, CsiRtUnitCtorName, &M);
  BasicBlock *CtorBB = BasicBlock::Create(C, "", Ctor);
  IRBuilder<> IRB(ReturnInst::Create(C, CtorBB));

  StructType *UnitFedTableType = getUnitFedTableType(C, FrontEndDataTable::getPointerType(C));

  // Lookup __csirt_unit_init
  SmallVector<Type *, 4> InitArgTypes({
      IRB.getInt8PtrTy(),
      PointerType::get(UnitFedTableType, 0),
      InitCallsiteToFunction->getType()
  });
  FunctionType *InitFunctionTy = FunctionType::get(IRB.getVoidTy(), InitArgTypes, false);
  Function *InitFunction = checkCsiInterfaceFunction(
      M.getOrInsertFunction(CsiRtUnitInitName, InitFunctionTy));
  assert(InitFunction);

  // Insert __csi_func_id_<f> weak symbols for all defined functions
  // and generate the runtime code that stores to all of them.
  generateInitCallsiteToFunction(M);

  SmallVector<Constant *, 4> UnitFedTables({
      fedTableToUnitFedTable(M, UnitFedTableType, BasicBlockFED),
      fedTableToUnitFedTable(M, UnitFedTableType, FunctionFED),
      fedTableToUnitFedTable(M, UnitFedTableType, FunctionExitFED),
      fedTableToUnitFedTable(M, UnitFedTableType, CallsiteFED),
      fedTableToUnitFedTable(M, UnitFedTableType, LoadFED),
      fedTableToUnitFedTable(M, UnitFedTableType, StoreFED),
  });

  ArrayType *UnitFedTableArrayType = ArrayType::get(UnitFedTableType, UnitFedTables.size());
  Constant *Table = ConstantArray::get(UnitFedTableArrayType, UnitFedTables);
  GlobalVariable *GV = new GlobalVariable(M, UnitFedTableArrayType, false, GlobalValue::InternalLinkage, Table, CsiUnitFedTableArrayName);

  Constant *Zero = ConstantInt::get(IRB.getInt32Ty(), 0);
  Value *GepArgs[] = {Zero, Zero};

  // Insert call to __csirt_unit_init
  CallInst *Call = IRB.CreateCall(InitFunction, {
      IRB.CreateGlobalStringPtr(M.getName()),
      ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs),
      InitCallsiteToFunction
  });

  // Add the constructor to the global list
  appendToGlobalCtors(M, Ctor, CsiUnitCtorPriority);

  CallGraphNode *CNCtor = CG->getOrInsertFunction(Ctor);
  CallGraphNode *CNFunc = CG->getOrInsertFunction(InitFunction);
  CNCtor->addCalledFunction(Call, CNFunc);
}

void ComprehensiveStaticInstrumentation::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<CallGraphWrapperPass>();
}

bool ComprehensiveStaticInstrumentation::shouldNotInstrumentFunction(Function &F) {
    Module &M = *F.getParent();
    if (F.hasName() && F.getName() == CsiRtUnitCtorName) {
        return true;
    }
    // Don't instrument functions that will run before or
    // simultaneously with CSI ctors.
    GlobalVariable *GV = M.getGlobalVariable("llvm.global_ctors");
    if (GV == nullptr) return false;
    ConstantArray *CA = cast<ConstantArray>(GV->getInitializer());
    for (Use &OP : CA->operands()) {
        if (isa<ConstantAggregateZero>(OP)) continue;
        ConstantStruct *CS = cast<ConstantStruct>(OP);

        if (Function *CF = dyn_cast<Function>(CS->getOperand(1))) {
            uint64_t Priority = dyn_cast<ConstantInt>(CS->getOperand(0))->getLimitedValue();
            if (Priority <= CsiUnitCtorPriority && CF->getName() == F.getName()) {
              // Do not instrument F.
              return true;
            }
        }
    }
    // false means do instrument it.
    return false;
}

void ComprehensiveStaticInstrumentation::computeAttributesForMemoryAccesses(
    SmallVectorImpl<std::pair<BasicBlock::iterator, csi_acc_prop_t> > &MemoryAccesses,
    SmallVectorImpl<BasicBlock::iterator> &LocalAccesses) {
  SmallSet<Value*, 8> WriteTargets;

  for (SmallVectorImpl<BasicBlock::iterator>::reverse_iterator It = LocalAccesses.rbegin(),
      E = LocalAccesses.rend(); It != E; ++It) {
    BasicBlock::iterator II = *It;
    Instruction *I = &(*II);
    if (StoreInst *Store = dyn_cast<StoreInst>(I)) {
      WriteTargets.insert(Store->getPointerOperand());
      MemoryAccesses.push_back(
        std::make_pair(II, csi_acc_prop_t{0, false, false, false}));
    } else {
      LoadInst *Load = cast<LoadInst>(I);
      Value *Addr = Load->getPointerOperand();
      bool HasBeenSeen = WriteTargets.count(Addr) > 0;
      MemoryAccesses.push_back(
        std::make_pair(II, csi_acc_prop_t{0, false, false, HasBeenSeen}));
    }
  }
  LocalAccesses.clear();
}

bool ComprehensiveStaticInstrumentation::runOnModule(Module &M) {
  initializeCsi(M);

  for (Function &F : M) {
    instrumentFunction(F);
  }

  finalizeCsi(M);
  return true;  // We always insert the unit constructor.
}

void ComprehensiveStaticInstrumentation::instrumentFunction(Function &F) {
  // This is required to prevent instrumenting the call to
  // __csi_module_init from within the module constructor.
  if (F.empty() || shouldNotInstrumentFunction(F)) {
    return;
  }

  SmallVector<std::pair<BasicBlock::iterator, csi_acc_prop_t>, 8> MemoryAccesses;
  SmallSet<Value*, 8> WriteTargets;
  SmallVector<BasicBlock::iterator, 8> LocalMemoryAccesses;

  SmallVector<BasicBlock::iterator, 8> RetVec;
  SmallVector<BasicBlock::iterator, 8> MemIntrinsics;
  SmallVector<BasicBlock::iterator, 8> Callsites;
  const DataLayout &DL = F.getParent()->getDataLayout();

  // Traverse all instructions in a function and insert instrumentation
  // on load & store
  for (BasicBlock &BB : F) {
    for (auto II = BB.begin(); II != BB.end(); II++) {
      Instruction *I = &(*II);
      if (isa<LoadInst>(*I) || isa<StoreInst>(*I)) {
        LocalMemoryAccesses.push_back(II);
      } else if (isa<ReturnInst>(*I)) {
        RetVec.push_back(II);
      } else if (isa<CallInst>(*I) || isa<InvokeInst>(*I)) {
        Callsites.push_back(II);
        if (isa<MemIntrinsic>(I))
          MemIntrinsics.push_back(II);
        computeAttributesForMemoryAccesses(MemoryAccesses, LocalMemoryAccesses);
      }
    }
    computeAttributesForMemoryAccesses(MemoryAccesses, LocalMemoryAccesses);
  }

  // Do this work in a separate loop after copying the iterators so that we
  // aren't modifying the list as we're iterating.
  for (std::pair<BasicBlock::iterator, csi_acc_prop_t> p : MemoryAccesses)
    instrumentLoadOrStore(p.first, p.second, DL);

  for (BasicBlock::iterator I : MemIntrinsics)
    instrumentMemIntrinsic(I);

  for (BasicBlock::iterator I : Callsites) {
    instrumentCallsite(I);
  }

  // Instrument basic blocks
  // Note that we do this before function entry so that we put this at the
  // beginning of the basic block, and then the function entry call goes before
  // the call to basic block entry.
  uint64_t LocalId = FunctionFED.add(F);
  FuncOffsetMap[F.getName()] = LocalId;
  for (BasicBlock &BB : F) {
    instrumentBasicBlock(BB);
  }

  // Instrument function entry/exit points.
  IRBuilder<> IRB(&*F.getEntryBlock().getFirstInsertionPt());

  Value *FuncId = FunctionFED.localToGlobalId(LocalId, IRB);
  IRB.CreateCall(CsiFuncEntry, {FuncId});

  for (BasicBlock::iterator I : RetVec) {
      Instruction *RetInst = &(*I);
      IRBuilder<> IRBRet(RetInst);
      uint64_t ExitLocalId = FunctionExitFED.add(F);
      Value *ExitCsiId = FunctionExitFED.localToGlobalId(ExitLocalId, IRBRet);
      IRBRet.CreateCall(CsiFuncExit, {ExitCsiId, FuncId});
  }
}
