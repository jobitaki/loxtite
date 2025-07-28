; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@format_str_0 = private constant [5 x i8] c"%.6g\0A"

define void @main() {
  %1 = call double @fibonacci(double 1.000000e+01)
  %2 = call i32 (ptr, ...) @printf(ptr @format_str_0, double %1)
  ret void
}

define double @fibonacci(double %0) {
  %2 = alloca double, i64 1, align 8
  %3 = insertvalue { ptr, ptr, i64 } undef, ptr %2, 0
  %4 = insertvalue { ptr, ptr, i64 } %3, ptr %2, 1
  %5 = insertvalue { ptr, ptr, i64 } %4, i64 0, 2
  %6 = extractvalue { ptr, ptr, i64 } %5, 1
  store double %0, ptr %6, align 8
  %7 = alloca double, i64 1, align 8
  %8 = insertvalue { ptr, ptr, i64 } undef, ptr %7, 0
  %9 = insertvalue { ptr, ptr, i64 } %8, ptr %7, 1
  %10 = insertvalue { ptr, ptr, i64 } %9, i64 0, 2
  %11 = extractvalue { ptr, ptr, i64 } %10, 1
  store double 0.000000e+00, ptr %11, align 8
  %12 = extractvalue { ptr, ptr, i64 } %5, 1
  %13 = load double, ptr %12, align 8
  %14 = fcmp oeq double %13, 0.000000e+00
  br i1 %14, label %15, label %17

15:                                               ; preds = %1
  %16 = extractvalue { ptr, ptr, i64 } %10, 1
  store double 0.000000e+00, ptr %16, align 8
  br label %35

17:                                               ; preds = %1
  %18 = extractvalue { ptr, ptr, i64 } %5, 1
  %19 = load double, ptr %18, align 8
  %20 = fcmp oeq double %19, 1.000000e+00
  br i1 %20, label %21, label %23

21:                                               ; preds = %17
  %22 = extractvalue { ptr, ptr, i64 } %10, 1
  store double 1.000000e+00, ptr %22, align 8
  br label %34

23:                                               ; preds = %17
  %24 = extractvalue { ptr, ptr, i64 } %5, 1
  %25 = load double, ptr %24, align 8
  %26 = fsub double %25, 1.000000e+00
  %27 = call double @fibonacci(double %26)
  %28 = extractvalue { ptr, ptr, i64 } %5, 1
  %29 = load double, ptr %28, align 8
  %30 = fsub double %29, 2.000000e+00
  %31 = call double @fibonacci(double %30)
  %32 = fadd double %27, %31
  %33 = extractvalue { ptr, ptr, i64 } %10, 1
  store double %32, ptr %33, align 8
  br label %34

34:                                               ; preds = %21, %23
  br label %35

35:                                               ; preds = %15, %34
  %36 = extractvalue { ptr, ptr, i64 } %10, 1
  %37 = load double, ptr %36, align 8
  ret double %37
}

declare i32 @printf(ptr, ...)
