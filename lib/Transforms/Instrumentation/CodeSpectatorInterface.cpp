#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h" // for itostr function
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

// see
// http://llvm.org/docs/ProgrammersManual.html#the-debug-macro-and-debug-option
// on how to use debugging infrastructure in LLVM
// also used by STATISTIC macro, so need to define this before using STATISTIC
#define DEBUG_TYPE "csi-func"

// XXX: Not sure how to turn these on yet
STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");
STATISTIC(NumAccessesWithBadSize, "Number of accesses with bad size");

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
const char *const CsiInitRelTablesName = "__csi_init_rel_tables";
const char *const CsiInitCallsiteToFunctionName = "__csi_init_callsite_to_function";

const uint64_t CsiCallsiteUnknownTargetId = 0xffffffffffffffff;
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
    uint64_t Id = add(llvm::getDISubprogram(&F));
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

  PointerType *getPointerType(LLVMContext &C) const {
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

  // Create a struct type to match the "struct fed_entry" defined in csirt.c
  StructType *getEntryStructType(LLVMContext &C) const {
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

class RelationTable {
public:
  typedef std::pair<uint64_t, uint64_t> Range;
  typedef std::map<Value *, uint64_t>::iterator iterator;
  typedef std::map<Value *, Range>::iterator range_iterator;

  RelationTable() {}

  size_t size() const { return relations.size() + rangeRelations.size(); }
  iterator begin() { return relations.begin(); }
  iterator end() { return relations.end(); }
  range_iterator range_begin() { return rangeRelations.begin(); }
  range_iterator range_end() { return rangeRelations.end(); }

  // Add a one-to-one relation of a Value (e.g. basic block) to a
  // local ID (e.g. the ID of its parent function)
  void addRelation(Value *V, uint64_t id) {
    assert(relations.find(V) == relations.end() && "Relation already exists.");
    assert(rangeRelations.find(V) == rangeRelations.end() && "Relation already exists.");
    relations[V] = id;
  }

  // Add a one-to-many relation of a value (e.g. function) to a range
  // of local IDs (e.g. the IDs of the basic blocks within the
  // function).
  void addRelation(Value *V, Range range) {
    assert(relations.find(V) == relations.end() && "Relation already exists.");
    assert(rangeRelations.find(V) == rangeRelations.end() && "Relation already exists.");
    rangeRelations[V] = range;
  }

  uint64_t get(Value *V) {
    assert(relations.find(V) != relations.end() && "Relation does not exist.");
    return relations[V];
  }

  Range getRange(Value *V) {
    assert(rangeRelations.find(V) != rangeRelations.end() && "Relation does not exist.");
    return rangeRelations[V];
  }

  // Return the range_t LLVM type
  static StructType *getRangeType(LLVMContext &C) {
    return StructType::get(IntegerType::get(C, 64),
                           IntegerType::get(C, 64),
                           nullptr);

  }

  // Return the CSIRT rel_table LLVM type
  static StructType *getTableType(LLVMContext &C) {
    return StructType::get(IntegerType::get(C, 64),
                           PointerType::get(IntegerType::get(C, 64), 0),
                           nullptr);
  }

  // Return the CSIRT rel_range_table LLVM type
  static StructType *getRangeTableType(LLVMContext &C) {
    return StructType::get(IntegerType::get(C, 64),
                           PointerType::get(getRangeType(C), 0),
                           nullptr);
  }

private:
  std::map<Value *, uint64_t> relations;
  std::map<Value *, Range> rangeRelations;
};

typedef struct {
  unsigned unused;
  bool unused2, unused3;
  bool read_before_write_in_bb;
} csi_acc_prop_t;

struct CodeSpectatorInterface : public ModulePass {
  static char ID;

  CodeSpectatorInterface() : ModulePass(ID) {}
  const char *getPassName() const override;
  bool doInitialization(Module &M) override;
  bool runOnModule(Module &M) override;
  bool runOnFunction(Function &F);
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  // not overriding doFinalization

private:
  int getNumBytesAccessed(Value *Addr, const DataLayout &DL);
  // initialize CSI instrumentation functions for load and store
  void initializeLoadStoreCallbacks(Module &M);
  // initialize CSI instrumentation functions for function entry and exit
  void initializeFuncCallbacks(Module &M);
  // Basic block entry and exit instrumentation
  void initializeBasicBlockCallbacks(Module &M);
  void initializeCallsiteCallbacks(Module &M);
  void initializeFEDTables(Module &M);
  void initializeRelTableFunctions(Module &M);
  void generateInitRelationTables(Module &M);
  void generateInitCallsiteToFunction(Module &M);

  // actually insert the instrumentation call
  bool instrumentLoadOrStore(BasicBlock::iterator Iter, csi_acc_prop_t prop, const DataLayout &DL);

  void computeAttributesForMemoryAccesses(
      SmallVectorImpl<std::pair<BasicBlock::iterator, csi_acc_prop_t> > &Accesses,
      SmallVectorImpl<BasicBlock::iterator> &LocalAccesses);

  bool addLoadStoreInstrumentation(BasicBlock::iterator Iter,
                                   Function *BeforeFn,
                                   Function *AfterFn,
                                   Value *CsiId,
                                   Type *AddrType,
                                   Value *Addr,
                                   int NumBytes,
                                   csi_acc_prop_t prop);

  FunctionType *getInitRelTableFunctionType(LLVMContext &C);
  // instrument a call to memmove, memcpy, or memset
  void instrumentMemIntrinsic(BasicBlock::iterator I);
  void instrumentCallsite(CallSite &CS);
  bool instrumentBasicBlock(BasicBlock &BB);
  bool FunctionCallsFunction(Function *F, Function *G);
  bool ShouldNotInstrumentFunction(Function &F);
  void InitializeCsi(Module &M);
  void FinalizeCsi(Module &M);

  CallGraph *CG;

  FrontEndDataTable FunctionFED, FunctionExitFED, BasicBlockFED,
    CallsiteFED, LoadFED, StoreFED;

  RelationTable BasicBlockToFunctionRelTable, FunctionToBasicBlocksRelTable;

  Function *CsiBeforeRead;
  Function *CsiAfterRead;
  Function *CsiBeforeWrite;
  Function *CsiAfterWrite;

  Function *CsiFuncEntry;
  Function *CsiFuncExit;
  Function *CsiBBEntry, *CsiBBExit;
  Function *MemmoveFn, *MemcpyFn, *MemsetFn;
  Function *CsiBeforeCallsite;

  Function *InitRelTables, *InitCallsiteToFunction;

  Type *IntptrTy;

  std::map<std::string, uint64_t> FuncOffsetMap;
}; //struct CodeSpectatorInterface
} // anonymous namespace

// the address matters but not the init value
char CodeSpectatorInterface::ID = 0;
INITIALIZE_PASS(CodeSpectatorInterface, "CSI-func", "CodeSpectatorInterface function pass",
                false, false)

const char *CodeSpectatorInterface::getPassName() const {
  return "CodeSpectatorInterface";
}

ModulePass *llvm::createCodeSpectatorInterfacePass() {
  return new CodeSpectatorInterface();
}

/**
 * initialize the declaration of function call instrumentation functions
 *
 * void __csi_func_entry(uint64_t csi_id, void *function, void *return_addr, char *func_name);
 * void __csi_func_exit(uint64_t csi_id, void *function, void *return_addr, char *func_name);
 */
void CodeSpectatorInterface::initializeFuncCallbacks(Module &M) {
  IRBuilder<> IRB(M.getContext());
  CsiFuncEntry = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_func_entry", IRB.getVoidTy(), IRB.getInt64Ty(), nullptr));
  CsiFuncExit = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_func_exit", IRB.getVoidTy(), IRB.getInt64Ty(), IRB.getInt64Ty(), nullptr));
}

void CodeSpectatorInterface::initializeBasicBlockCallbacks(Module &M) {
  IRBuilder<> IRB(M.getContext());
  SmallVector<Type *, 4> ArgTypes({IRB.getInt64Ty()});
  FunctionType *FnType = FunctionType::get(IRB.getVoidTy(), ArgTypes, false);
  CsiBBEntry = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_bb_entry", FnType));

  CsiBBExit = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_bb_exit", FnType));
}

void CodeSpectatorInterface::initializeCallsiteCallbacks(Module &M) {
  IRBuilder<> IRB(M.getContext());
  SmallVector<Type *, 4> ArgTypes({IRB.getInt64Ty(), IRB.getInt64Ty()});
  FunctionType *FnType = FunctionType::get(IRB.getVoidTy(), ArgTypes, false);
  CsiBeforeCallsite = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_callsite", FnType));
}

/**
 * initialize the declaration of instrumentation functions
 *
 * void __csi_before_load(uint64_t csi_id, void *addr, uint32_t num_bytes, uint64_t prop);
 *
 * where num_bytes = 1, 2, 4, 8.
 *
 * Presumably aligned / unaligned accesses are specified by the attr
 */
void CodeSpectatorInterface::initializeLoadStoreCallbacks(Module &M) {

  IRBuilder<> IRB(M.getContext());
  Type *RetType = IRB.getVoidTy();            // return void
  Type *AddrType = IRB.getInt8PtrTy();        // void *addr
  Type *NumBytesType = IRB.getInt32Ty();      // int num_bytes

  // Initialize the instrumentation for reads, writes

  // void __csi_before_load(uint64_t csi_id, void *addr, int num_bytes, int attr);
  CsiBeforeRead = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_load", RetType,
        IRB.getInt64Ty(), AddrType, NumBytesType, IRB.getInt64Ty(), nullptr));

  // void __csi_after_load(uint64_t csi_id, void *addr, int num_bytes, int attr);
  SmallString<32> AfterReadName("__csi_after_load");
  CsiAfterRead = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_load", RetType,
        IRB.getInt64Ty(), AddrType, NumBytesType, IRB.getInt64Ty(), nullptr));

  // void __csi_before_store(uint64_t csi_id, void *addr, int num_bytes, int attr);
  CsiBeforeWrite = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_store", RetType,
        IRB.getInt64Ty(), AddrType, NumBytesType, IRB.getInt64Ty(), nullptr));

  // void __csi_after_store(uint64_t csi_id, void *addr, int num_bytes, int attr);
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

FunctionType *CodeSpectatorInterface::getInitRelTableFunctionType(LLVMContext &C) {
  // This must match the definition of __csi_init_rel_tables_t in csirt.c.
  // typedef void (*__csi_init_rel_tables_t)(rel_table *rel_bb_to_func, rel_range_table *rel_func_to_bb);
  SmallVector<Type *, 2> ArgTypes({
      PointerType::get(RelationTable::getTableType(C), 0),
      PointerType::get(RelationTable::getRangeTableType(C), 0)
  });

  return FunctionType::get(Type::getVoidTy(C), ArgTypes, false);
}

int CodeSpectatorInterface::getNumBytesAccessed(Value *Addr,
                                                const DataLayout &DL) {
  Type *OrigPtrTy = Addr->getType();
  Type *OrigTy = cast<PointerType>(OrigPtrTy)->getElementType();
  assert(OrigTy->isSized());
  uint32_t TypeSize = DL.getTypeStoreSizeInBits(OrigTy);
  if (TypeSize != 8  && TypeSize != 16 && TypeSize != 32 && TypeSize != 64 && TypeSize != 128) {
    DEBUG_WITH_TYPE("csi-func",
        errs() << "Bad size " << TypeSize << " at addr " << Addr << "\n");
    NumAccessesWithBadSize++;
    return -1;
  }
  return TypeSize / 8;
}

bool CodeSpectatorInterface::addLoadStoreInstrumentation(BasicBlock::iterator Iter,
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

  return true;
}

bool CodeSpectatorInterface::instrumentLoadOrStore(BasicBlock::iterator Iter,
                                                   csi_acc_prop_t prop,
                                                   const DataLayout &DL) {

  DEBUG_WITH_TYPE("csi-func",
      errs() << "CSI_func: instrument instruction " << *Iter << "\n");

  Instruction *I = &(*Iter);
  // takes pointer to Instruction and inserts before the instruction
  IRBuilder<> IRB(&(*Iter));
  bool IsWrite = isa<StoreInst>(I);
  Value *Addr = IsWrite ?
      cast<StoreInst>(I)->getPointerOperand()
      : cast<LoadInst>(I)->getPointerOperand();

  int NumBytes = getNumBytesAccessed(Addr, DL);
  Type *AddrType = IRB.getInt8PtrTy();

  if (NumBytes == -1) return false; // size that we don't recognize

  bool Res = false;

  if(IsWrite) {
    uint64_t LocalId = StoreFED.add(*I);
    Value *CsiId = StoreFED.localToGlobalId(LocalId, IRB);
    Res = addLoadStoreInstrumentation(
        Iter, CsiBeforeWrite, CsiAfterWrite, CsiId, AddrType, Addr, NumBytes, prop);
    NumInstrumentedWrites++;

  } else { // is read
    uint64_t LocalId = LoadFED.add(*I);
    Value *CsiId = LoadFED.localToGlobalId(LocalId, IRB);
    Res = addLoadStoreInstrumentation(
        Iter, CsiBeforeRead, CsiAfterRead, CsiId, AddrType, Addr, NumBytes, prop);
    NumInstrumentedReads++;
  }

  return Res;
}

// If a memset intrinsic gets inlined by the code gen, we will miss races on it.
// So, we either need to ensure the intrinsic is not inlined, or instrument it.
// We do not instrument memset/memmove/memcpy intrinsics (too complicated),
// instead we simply replace them with regular function calls, which are then
// intercepted by the run-time.
// Since our pass runs after everyone else, the calls should not be
// replaced back with intrinsics. If that becomes wrong at some point,
// we will need to call e.g. __csi_memset to avoid the intrinsics.
void CodeSpectatorInterface::instrumentMemIntrinsic(BasicBlock::iterator Iter) {
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

bool CodeSpectatorInterface::instrumentBasicBlock(BasicBlock &BB) {
  IRBuilder<> IRB(BB.getFirstInsertionPt());
  uint64_t LocalId = BasicBlockFED.add(BB);
  Value *CsiId = BasicBlockFED.localToGlobalId(LocalId, IRB);

  IRB.CreateCall(CsiBBEntry, {CsiId});

  TerminatorInst *TI = BB.getTerminator();
  IRB.SetInsertPoint(TI);
  IRB.CreateCall(CsiBBExit, {CsiId});

  BasicBlockToFunctionRelTable.addRelation(&BB, FunctionFED.getId(BB.getParent()));
  return true;
}

void CodeSpectatorInterface::instrumentCallsite(CallSite &CS) {
  Instruction *I = CS.getInstruction();
  Module *M = I->getParent()->getParent()->getParent();
  Function *Called = CS.getCalledFunction();

  if (Called && Called->getName().startswith("llvm.dbg")) {
      return;
  }

  IRBuilder<> IRB(I);
  uint64_t LocalId = CallsiteFED.add(*I);
  Value *CsiId = CallsiteFED.localToGlobalId(LocalId, IRB);

  std::string GVName = CsiFuncIdVariablePrefix + Called->getName().str();
  GlobalVariable *FuncIdGV = dyn_cast<GlobalVariable>(M->getOrInsertGlobal(GVName, IRB.getInt64Ty()));
  assert(FuncIdGV);
  FuncIdGV->setConstant(false);
  FuncIdGV->setLinkage(GlobalValue::WeakAnyLinkage);
  FuncIdGV->setInitializer(IRB.getInt64(CsiCallsiteUnknownTargetId));

  Value *FuncId = IRB.CreateLoad(FuncIdGV);
  IRB.CreateCall(CsiBeforeCallsite, {CsiId, FuncId});
}

bool CodeSpectatorInterface::doInitialization(Module &M) {
  DEBUG_WITH_TYPE("csi-func", errs() << "CSI_func: doInitialization" << "\n");

  IntptrTy = M.getDataLayout().getIntPtrType(M.getContext());

  DEBUG_WITH_TYPE("csi-func",
      errs() << "CSI_func: doInitialization done" << "\n");
  return true;
}

void CodeSpectatorInterface::initializeFEDTables(Module &M) {
  FunctionFED = FrontEndDataTable(M, CsiFunctionBaseIdName);
  FunctionExitFED = FrontEndDataTable(M, CsiFunctionExitBaseIdName);
  BasicBlockFED = FrontEndDataTable(M, CsiBasicBlockBaseIdName);
  CallsiteFED = FrontEndDataTable(M, CsiCallsiteBaseIdName);
  LoadFED = FrontEndDataTable(M, CsiLoadBaseIdName);
  StoreFED = FrontEndDataTable(M, CsiStoreBaseIdName);
}

void CodeSpectatorInterface::initializeRelTableFunctions(Module &M) {
  FunctionType *FnType = getInitRelTableFunctionType(M.getContext());
  InitRelTables = checkCsiInterfaceFunction(M.getOrInsertFunction(CsiInitRelTablesName, FnType));
  assert(InitRelTables);
  InitRelTables->setLinkage(GlobalValue::InternalLinkage);

  FnType = FunctionType::get(Type::getVoidTy(M.getContext()), {}, false);
  InitCallsiteToFunction = checkCsiInterfaceFunction(M.getOrInsertFunction(CsiInitCallsiteToFunctionName, FnType));
  assert(InitCallsiteToFunction);
  InitCallsiteToFunction->setLinkage(GlobalValue::InternalLinkage);
}

void CodeSpectatorInterface::generateInitRelationTables(Module &M) {
  LLVMContext &C = M.getContext();
  BasicBlock *EntryBB = BasicBlock::Create(C, "", InitRelTables);
  IRBuilder<> IRB(ReturnInst::Create(C, EntryBB));

  GlobalVariable *FuncBaseGV = FunctionFED.baseId(),
    *BBBaseGV = BasicBlockFED.baseId();
  LoadInst *FuncBaseId = IRB.CreateLoad(FuncBaseGV),
    *BBBaseId = IRB.CreateLoad(BBBaseGV);

  Function::ArgumentListType::iterator ArgIter = InitRelTables->getArgumentList().begin();
  Argument *RelTablePtr = &(*ArgIter++);
  Argument *RelRangeTablePtr = &(*ArgIter++);
  assert(RelTablePtr && RelRangeTablePtr);

  // Load the pointer to the id array from the argument (which is a struct pointer).
  SmallVector<Value *, 2> Index({ IRB.getInt32(0), // Dereference struct pointer
        IRB.getInt32(1) });                        // Get address of second field
  Value *RelTableIdArray = IRB.CreateLoad(IRB.CreateInBoundsGEP(RelTablePtr, Index));
  for (RelationTable::iterator it = BasicBlockToFunctionRelTable.begin(),
         ite = BasicBlockToFunctionRelTable.end(); it != ite; ++it) {
    BasicBlock *BB = dyn_cast<BasicBlock>(it->first);
    assert(BB);
    uint64_t bbid = BasicBlockFED.getId(BB), funcid = FunctionFED.getId(BB->getParent());
    Value *GlobalBBId = IRB.CreateAdd(BBBaseId, IRB.getInt64(bbid));
    Value *GlobalFuncId = IRB.CreateAdd(FuncBaseId, IRB.getInt64(funcid));
    // Get the address of the correct element in the id array and store to it.
    SmallVector<Value *, 1> ArrayIndex({ GlobalBBId });
    Value *IdPtr = IRB.CreateInBoundsGEP(RelTableIdArray, ArrayIndex);
    IRB.CreateStore(GlobalFuncId, IdPtr);
  }

  // Repeat the same for the range table
  Value *RelRangeTableRangeArray = IRB.CreateLoad(IRB.CreateInBoundsGEP(RelRangeTablePtr, Index));
  for (RelationTable::range_iterator it = FunctionToBasicBlocksRelTable.range_begin(),
         ite = FunctionToBasicBlocksRelTable.range_end(); it != ite; ++it) {
    Function *F = dyn_cast<Function>(it->first);
    assert(F);
    uint64_t funcid = FunctionFED.getId(F);
    uint64_t firstBBId = it->second.first, lastBBId = it->second.second;
    Value *GlobalFuncId = IRB.CreateAdd(FuncBaseId, IRB.getInt64(funcid));
    Value *GlobalStartBBId = IRB.CreateAdd(BBBaseId, IRB.getInt64(firstBBId)),
      *GlobalLastBBId = IRB.CreateAdd(BBBaseId, IRB.getInt64(lastBBId));

    // Create the range_t instance.
    SmallVector<Constant *, 2> Undefs({UndefValue::get(IntegerType::get(C, 64)), UndefValue::get(IntegerType::get(C, 64))});
    Constant *RangeStruct = ConstantStruct::getAnon(Undefs);
    Value *GlobalBBRange = IRB.CreateInsertValue(RangeStruct, GlobalStartBBId, {0});
    GlobalBBRange = IRB.CreateInsertValue(GlobalBBRange, GlobalLastBBId, {1});

    // Get the address of the correct element in the range array and store to it.
    SmallVector<Value *, 1> ArrayIndex({ GlobalFuncId });
    Value *RangePtr = IRB.CreateInBoundsGEP(RelRangeTableRangeArray, ArrayIndex);
    IRB.CreateStore(GlobalBBRange, RangePtr);
  }
}

void CodeSpectatorInterface::generateInitCallsiteToFunction(Module &M) {
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

void CodeSpectatorInterface::InitializeCsi(Module &M) {
  initializeFEDTables(M);
  initializeFuncCallbacks(M);
  initializeLoadStoreCallbacks(M);
  initializeBasicBlockCallbacks(M);
  initializeCallsiteCallbacks(M);
  initializeRelTableFunctions(M);

  CG = &getAnalysis<CallGraphWrapperPass>().getCallGraph();
}

void CodeSpectatorInterface::FinalizeCsi(Module &M) {
  LLVMContext &C = M.getContext();

  // Add CSI global constructor, which calls unit init.
  Function *Ctor = Function::Create(
      FunctionType::get(Type::getVoidTy(C), false),
      GlobalValue::InternalLinkage, CsiRtUnitCtorName, &M);
  BasicBlock *CtorBB = BasicBlock::Create(C, "", Ctor);
  IRBuilder<> IRB(ReturnInst::Create(C, CtorBB));

  // Lookup __csirt_unit_init
  SmallVector<Type *, 4> InitArgTypes({
      IRB.getInt8PtrTy(),
      IRB.getInt64Ty(),
      PointerType::get(IRB.getInt64Ty(), 0),
      FunctionFED.getPointerType(C),
      IRB.getInt64Ty(),
      PointerType::get(IRB.getInt64Ty(), 0),
      FunctionExitFED.getPointerType(C),
      IRB.getInt64Ty(),
      PointerType::get(IRB.getInt64Ty(), 0),
      BasicBlockFED.getPointerType(C),
      IRB.getInt64Ty(),
      PointerType::get(IRB.getInt64Ty(), 0),
      CallsiteFED.getPointerType(C),
      IRB.getInt64Ty(),
      PointerType::get(IRB.getInt64Ty(), 0),
      LoadFED.getPointerType(C),
      IRB.getInt64Ty(),
      PointerType::get(IRB.getInt64Ty(), 0),
      StoreFED.getPointerType(C),
      IRB.getInt64Ty(),
      IRB.getInt64Ty(),
      InitRelTables->getType(),
      InitCallsiteToFunction->getType()
  });
  FunctionType *InitFunctionTy = FunctionType::get(IRB.getVoidTy(), InitArgTypes, false);
  Function *InitFunction = checkCsiInterfaceFunction(
      M.getOrInsertFunction(CsiRtUnitInitName, InitFunctionTy));
  assert(InitFunction);

  // Insert __csi_func_id_<f> weak symbols for all defined functions
  // and generate the runtime code that stores to all of them.
  generateInitCallsiteToFunction(M);

  // Generate the function body to initialize the relation tables.
  generateInitRelationTables(M);

  Constant *FunctionFEDPtr = FunctionFED.insertIntoModule(M),
    *FunctionExitFEDPtr = FunctionExitFED.insertIntoModule(M),
    *BasicBlockFEDPtr = BasicBlockFED.insertIntoModule(M),
    *CallsiteFEDPtr = CallsiteFED.insertIntoModule(M),
    *LoadFEDPtr = LoadFED.insertIntoModule(M),
    *StoreFEDPtr = StoreFED.insertIntoModule(M);

  // Insert call to __csirt_unit_init
  CallInst *Call = IRB.CreateCall(InitFunction, {
      IRB.CreateGlobalStringPtr(M.getName()),
      IRB.getInt64(FunctionFED.size()),
      FunctionFED.baseId(),
      FunctionFEDPtr,
      IRB.getInt64(FunctionExitFED.size()),
      FunctionExitFED.baseId(),
      FunctionExitFEDPtr,
      IRB.getInt64(BasicBlockFED.size()),
      BasicBlockFED.baseId(),
      BasicBlockFEDPtr,
      IRB.getInt64(CallsiteFED.size()),
      CallsiteFED.baseId(),
      CallsiteFEDPtr,
      IRB.getInt64(LoadFED.size()),
      LoadFED.baseId(),
      LoadFEDPtr,
      IRB.getInt64(StoreFED.size()),
      StoreFED.baseId(),
      StoreFEDPtr,
      IRB.getInt64(BasicBlockToFunctionRelTable.size()),
      IRB.getInt64(FunctionToBasicBlocksRelTable.size()),
      InitRelTables,
      InitCallsiteToFunction
  });

  // Add the constructor to the global list
  appendToGlobalCtors(M, Ctor, CsiUnitCtorPriority);

  CallGraphNode *CNCtor = CG->getOrInsertFunction(Ctor);
  CallGraphNode *CNFunc = CG->getOrInsertFunction(InitFunction);
  CNCtor->addCalledFunction(Call, CNFunc);
}

void CodeSpectatorInterface::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<CallGraphWrapperPass>();
}

// Recursively determine if F calls G. Return true if so. Conservatively, if F makes
// any internal indirect function calls, assume it calls G.
bool CodeSpectatorInterface::FunctionCallsFunction(Function *F, Function *G) {
  assert(F && G && CG);
  CallGraphNode *CGN = (*CG)[F];
  // Assume external functions cannot make calls to internal functions.
  if (!F->hasLocalLinkage() && G->hasLocalLinkage()) return false;
  // Assume function declarations won't make calls to internal
  // functions. TODO: This may not be correct in general.
  if (F->isDeclaration()) return false;
  for (CallGraphNode::iterator it = CGN->begin(), ite = CGN->end(); it != ite; ++it) {
    Function *Called = it->second->getFunction();
    if (Called == NULL) {
      // Indirect call
      return true;
    } else if (Called == G) {
      return true;
    } else if (G->hasLocalLinkage() && !Called->hasLocalLinkage()) {
      // Assume external functions cannot make calls to internal functions.
      continue;
    }
  }
  for (CallGraphNode::iterator it = CGN->begin(), ite = CGN->end(); it != ite; ++it) {
    Function *Called = it->second->getFunction();
    if (FunctionCallsFunction(Called, G)) return true;
  }
  return false;
}

bool CodeSpectatorInterface::ShouldNotInstrumentFunction(Function &F) {
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
            if (Priority <= CsiUnitCtorPriority) {
                return CF->getName() == F.getName() ||  FunctionCallsFunction(CF, &F);
            }
        }
    }
    // false means do instrument it.
    return false;
}

void CodeSpectatorInterface::computeAttributesForMemoryAccesses(
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

bool CodeSpectatorInterface::runOnModule(Module &M) {
  InitializeCsi(M);

  for (Function &F : M)
    runOnFunction(F);

  FinalizeCsi(M);
  return true;  // we always insert the unit constructor
}

bool CodeSpectatorInterface::runOnFunction(Function &F) {
  // This is required to prevent instrumenting the call to
  // __csi_module_init from within the module constructor.
  if (F.empty() || ShouldNotInstrumentFunction(F)) {
      return false;
  }

  DEBUG_WITH_TYPE("csi-func",
                  errs() << "CSI_func: run on function " << F.getName() << "\n");

  SmallVector<std::pair<BasicBlock::iterator, csi_acc_prop_t>, 8> MemoryAccesses;
  SmallSet<Value*, 8> WriteTargets;
  SmallVector<BasicBlock::iterator, 8> LocalMemoryAccesses;

  SmallVector<BasicBlock::iterator, 8> RetVec;
  SmallVector<BasicBlock::iterator, 8> MemIntrinsics;
  SmallVector<BasicBlock::iterator, 8> Callsites;
  bool Modified = false;
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
    Modified |= instrumentLoadOrStore(p.first, p.second, DL);

  for (BasicBlock::iterator I : MemIntrinsics)
    instrumentMemIntrinsic(I);

  for (BasicBlock::iterator I : Callsites) {
    CallSite CS(I);
    instrumentCallsite(CS);
  }

  // Instrument basic blocks
  // Note that we do this before function entry so that we put this at the
  // beginning of the basic block, and then the function entry call goes before
  // the call to basic block entry.
  uint64_t LocalId = FunctionFED.add(F);
  FuncOffsetMap[F.getName()] = LocalId;
  for (BasicBlock &BB : F) {
    Modified |= instrumentBasicBlock(BB);
  }
  uint64_t BBStartId = BasicBlockFED.getId(&F.front()),
      BBEndId = BasicBlockFED.getId(&F.back());
  FunctionToBasicBlocksRelTable.addRelation(&F, std::make_pair(BBStartId, BBEndId));

  // Instrument function entry/exit points.
  IRBuilder<> IRB(F.getEntryBlock().getFirstInsertionPt());

  Value *FuncId = FunctionFED.localToGlobalId(LocalId, IRB);
  IRB.CreateCall(CsiFuncEntry, {FuncId});

  for (BasicBlock::iterator I : RetVec) {
      Instruction *RetInst = &(*I);
      IRBuilder<> IRBRet(RetInst);
      uint64_t ExitLocalId = FunctionExitFED.add(F);
      Value *ExitCsiId = FunctionExitFED.localToGlobalId(ExitLocalId, IRBRet);
      IRBRet.CreateCall(CsiFuncExit, {ExitCsiId, FuncId});
  }
  Modified = true;

  if(Modified) {
    DEBUG_WITH_TYPE("csi-func",
        errs() << "CSI_func: modified function " << F.getName() << "\n");
  }
  return Modified;
}

// End of compile-time pass
// ------------------------------------------------------------------------
