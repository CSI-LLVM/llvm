; Test CSI before/after callsite instrumentation.
;
; RUN: opt < %s -csi -S | FileCheck %s

; CHECK: @__csi_unit_callsite_base_id = internal global i64 0
; CHECK: @__csi_disable_instrumentation = external thread_local externally_initialized global i1
; CHECK: @__csi_func_id_foo = weak global i64 -1

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
; CHECK: %8 = load i64, i64* @__csi_unit_callsite_base_id
; CHECK-NEXT: %9 = add i64 %8, 0
; CHECK-NEXT: %10 = load i64, i64* @__csi_func_id_foo
; CHECK-NEXT: %11 = load i1, i1* @__csi_disable_instrumentation
; CHECK-NEXT: %12 = icmp eq i1 %11, false
; CHECK-NEXT: br i1 %12, label %13, label %14

; CHECK: <label>:13:
; CHECK-NEXT: store i1 true, i1* @__csi_disable_instrumentation
; CHECK-NEXT: call void @__csi_before_call(i64 %9, i64 %10, i64 0)
; CHECK-NEXT: store i1 false, i1* @__csi_disable_instrumentation
; CHECK: br label %14

; CHECK: <label>:14:
; CHECK-NEXT: %call = call i32 @foo()
; CHECK-NEXT: %15 = load i1, i1* @__csi_disable_instrumentation
; CHECK-NEXT: %16 = icmp eq i1 %15, false
; CHECK: br i1 %16, label %17, label %18

; CHECK: <label>:17:
; CHECK-NEXT: store i1 true, i1* @__csi_disable_instrumentation
; CHECK-NEXT: call void @__csi_after_call(i64 %9, i64 %10, i64 0)
; CHECK-NEXT: store i1 false, i1* @__csi_disable_instrumentation
; CHECK-NEXT: br label %18

; CHECK: <label>:18:
; CHECK: ret i32 %call

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Top-level:

; CHECK: declare void @__csi_before_call(i64, i64, i64)
; CHECK: declare void @__csi_after_call(i64, i64, i64)
