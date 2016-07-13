; Test CSI interface declarations.
;
; RUN: opt < %s -csi -S | FileCheck %s

define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %call = call i32 @foo()
  ret i32 %call
}

define internal i32 @foo() #0 {
entry:
  ret i32 1
}

; CHECK: @0 = private unnamed_addr constant [8 x i8] c"<stdin>\00"
; CHECK: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @csi.unit_ctor, i8* null }]

; CHECK: define internal void @csi.unit_ctor()
; CHECK-NEXT: call void @__csirt_unit_init(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @0, i32 0, i32 0))
; CHECK-NEXT: ret void

; CHECK: declare void @__csirt_unit_init(i8*)
