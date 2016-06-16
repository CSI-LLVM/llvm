; Test CSI function entry/exit instrumentation.
;
; RUN: opt < %s -csi -S | FileCheck %s

; CHECK: @__csi_unit_bb_base_id = internal global i64 0

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
; CHECK: %2 = load i64, i64* @__csi_unit_bb_base_id
; CHECK-NEXT: %3 = add i64 %2, 0
; CHECK-NEXT: call void @__csi_bb_entry(i64 %3)
; CHECK: %retval = alloca i32, align 4
; CHECK: store i32 0, i32* %retval, align 4
; CHECK: %call = call i32 @foo()
; CHECK: call void @__csi_bb_exit(i64 %3)
; CHECK: ret i32 %call

; CHECK: define internal i32 @foo()
; CHECK-NEXT: entry:
; CHECK: %2 = load i64, i64* @__csi_unit_bb_base_id
; CHECK: %3 = add i64 %2, 1
; CHECK-NEXT: call void @__csi_bb_entry(i64 %3)
; CHECK-NEXT: call void @__csi_bb_exit(i64 %3)
; CHECK: ret i32 1


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Top-level:

; CHECK: declare void @__csi_bb_entry(i64)
; CHECK: declare void @__csi_bb_exit(i64)
