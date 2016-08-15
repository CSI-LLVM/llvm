; Test CSI function entry/exit instrumentation.
;
; RUN: opt < %s -csi -S | FileCheck %s

; CHECK: @__csi_unit_func_base_id = internal global i64 0
; CHECK: @__csi_unit_func_exit_base_id = internal global i64 0

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

; CHECK: define i32 @main()
; CHECK-NEXT: entry:
; CHECK-NEXT: %0 = load i64, i64* @__csi_unit_func_base_id
; CHECK-NEXT: %1 = add i64 %0, 0
; CHECK-NEXT: call void @__csi_func_entry(i64 %1, i64 0)
; CHECK: %retval = alloca i32, align 4
; CHECK: store i32 0, i32* %retval, align 4
; CHECK: %call = call i32 @foo()
; CHECK: %19 = load i64, i64* @__csi_unit_func_exit_base_id
; CHECK-NEXT: %20 = add i64 %19, 0
; CHECK-NEXT: call void @__csi_func_exit(i64 %20, i64 %1, i64 0)
; CHECK-NEXT: ret i32 %call

; CHECK: define internal i32 @foo()
; CHECK-NEXT: entry:
; CHECK-NEXT: %0 = load i64, i64* @__csi_unit_func_base_id
; CHECK-NEXT: %1 = add i64 %0, 1
; CHECK-NEXT: call void @__csi_func_entry(i64 %1, i64 0)
; CHECK: %4 = load i64, i64* @__csi_unit_func_exit_base_id
; CHECK-NEXT: %5 = add i64 %4, 1
; CHECK-NEXT: call void @__csi_func_exit(i64 %5, i64 %1, i64 0)
; CHECK-NEXT: ret i32 1


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Top-level:

; CHECK: declare void @__csi_func_entry(i64, i64)
; CHECK: declare void @__csi_func_exit(i64, i64, i64)
