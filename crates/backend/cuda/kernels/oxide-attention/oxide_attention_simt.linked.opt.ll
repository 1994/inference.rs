; ModuleID = '/home/r/benchmarks/inference.rs/cuda-oxide-gdn-2026-10-15/source/crates/backend/cuda/kernels/oxide-attention/oxide_attention_simt.linked.ll'
source_filename = "llvm-link"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@__shared_mem_23b72fdefd237786_2 = local_unnamed_addr addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_23b72fdefd237786_1 = local_unnamed_addr addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_23b72fdefd237786_0 = local_unnamed_addr addrspace(3) global [4 x float] zeroinitializer, align 4
@.str = private unnamed_addr constant [11 x i8] c"__CUDA_FTZ\00", align 1
@llvm.used = appending global [1 x ptr] [ptr @attention_fp8_256], section "llvm.metadata"

; Function Attrs: convergent nounwind
define ptx_kernel void @attention_fp8_256(ptr nofree readonly captures(none) %v0, ptr nofree readonly captures(none) %v1, ptr nofree readonly captures(none) %v2, ptr nofree readonly captures(none) %v3, ptr nofree readonly captures(none) %v4, ptr nofree writeonly captures(none) %v5, i32 %v6, i32 %v7, float %v8, float %v9) #0 {
entry:
  %v20 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #8
  %v21 = zext nneg i32 %v20 to i64
  %v2210 = lshr i64 %v21, 5
  %v23 = and i64 %v21, 31
  %v24 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #8
  %v25 = zext nneg i32 %v24 to i64
  %v26 = udiv i32 %v24, 24
  %0 = mul i32 %v26, 24
  %v2767.decomposed = sub i32 %v24, %0
  %v28 = load i32, ptr %v3, align 4
  %v30 = add i32 %v28, %v26
  %v31 = getelementptr inbounds nuw i8, ptr %v3, i64 8
  %v32 = load i32, ptr %v31, align 4
  %v33 = add i32 %v30, %v32
  %v34 = getelementptr inbounds nuw i8, ptr %v3, i64 4
  %v35 = load i32, ptr %v34, align 4
  %v36.not = icmp slt i32 %v26, %v35
  %v38 = icmp sgt i32 %v33, -1
  %or.cond = select i1 %v36.not, i1 %v38, i1 false
  br i1 %or.cond, label %bb5, label %bb4

bb4:                                              ; preds = %entry
  %.idx = shl nuw nsw i64 %v25, 10
  %1 = getelementptr inbounds nuw i8, ptr %v5, i64 %.idx
  %v42 = getelementptr inbounds nuw [4 x i8], ptr %1, i64 %v21
  store float 0.000000e+00, ptr %v42, align 4
  br label %bb80

bb5:                                              ; preds = %entry
  %v45 = add nuw i32 %v33, 1
  %v46.not = icmp sgt i32 %v45, %v7
  %v48 = sub i32 %v45, %v7
  %v43.inv = icmp sgt i32 %v7, 0
  %2 = select i1 %v43.inv, i1 %v46.not, i1 false
  %v49 = select i1 %2, i32 %v48, i32 0
  %v50 = and i64 %v21, 15
  %v51 = shl nuw nsw i64 %v25, 8
  %v52 = shl nuw nsw i64 %v50, 4
  %3 = getelementptr inbounds nuw [4 x i8], ptr %v0, i64 %v51
  %v54 = getelementptr inbounds nuw [4 x i8], ptr %3, i64 %v52
  %v55 = load float, ptr %v54, align 4
  %v57 = getelementptr inbounds nuw i8, ptr %v54, i64 4
  %v58 = load float, ptr %v57, align 4
  %v60 = getelementptr inbounds nuw i8, ptr %v54, i64 8
  %v61 = load float, ptr %v60, align 4
  %v63 = getelementptr inbounds nuw i8, ptr %v54, i64 12
  %v64 = load float, ptr %v63, align 4
  %v66 = getelementptr inbounds nuw i8, ptr %v54, i64 16
  %v67 = load float, ptr %v66, align 4
  %v69 = getelementptr inbounds nuw i8, ptr %v54, i64 20
  %v70 = load float, ptr %v69, align 4
  %v72 = getelementptr inbounds nuw i8, ptr %v54, i64 24
  %v73 = load float, ptr %v72, align 4
  %v75 = getelementptr inbounds nuw i8, ptr %v54, i64 28
  %v76 = load float, ptr %v75, align 4
  %v78 = getelementptr inbounds nuw i8, ptr %v54, i64 32
  %v79 = load float, ptr %v78, align 4
  %v81 = getelementptr inbounds nuw i8, ptr %v54, i64 36
  %v82 = load float, ptr %v81, align 4
  %v84 = getelementptr inbounds nuw i8, ptr %v54, i64 40
  %v85 = load float, ptr %v84, align 4
  %v87 = getelementptr inbounds nuw i8, ptr %v54, i64 44
  %v88 = load float, ptr %v87, align 4
  %v90 = getelementptr inbounds nuw i8, ptr %v54, i64 48
  %v91 = load float, ptr %v90, align 4
  %v93 = getelementptr inbounds nuw i8, ptr %v54, i64 52
  %v94 = load float, ptr %v93, align 4
  %v96 = getelementptr inbounds nuw i8, ptr %v54, i64 56
  %v97 = load float, ptr %v96, align 4
  %v99 = getelementptr inbounds nuw i8, ptr %v54, i64 60
  %v100 = load float, ptr %v99, align 4
  %v101 = icmp eq i32 %v20, 0
  br i1 %v101, label %bb10, label %bb13

bb10:                                             ; preds = %bb5
  store float -1.000000e+30, ptr addrspace(3) @__shared_mem_23b72fdefd237786_0, align 4
  store float 0.000000e+00, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_0, i64 4), align 4
  br label %bb13

bb13:                                             ; preds = %bb10, %bb5
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #8
  %v105 = sdiv i32 %v49, 32
  %v10811 = lshr i32 %v33, 5
  %invariant.gep70 = getelementptr i8, ptr %v1, i64 %v52
  %v110.not72 = icmp sgt i32 %v105, %v10811
  br i1 %v110.not72, label %bb78, label %bb16.lr.ph

bb16.lr.ph:                                       ; preds = %bb13
  %.lhs.trunc = trunc nuw nsw i32 %v2767.decomposed to i8
  %4 = udiv i8 %.lhs.trunc, 6
  %.zext = zext nneg i8 %4 to i64
  %v117 = zext i32 %v6 to i64
  %v118 = shl nuw nsw i64 %.zext, 8
  %5 = mul nuw nsw i64 %v118, %v117
  %v122 = shl nuw nsw i64 %v2210, 1
  %v12312 = lshr i64 %v23, 4
  %v124 = or disjoint i64 %v122, %v12312
  %v12668 = icmp samesign ugt i32 %v20, 511
  %v1283 = icmp eq i64 %v50, 0
  %v156 = icmp eq i64 %v2210, 0
  %v157 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @__shared_mem_23b72fdefd237786_1, i64 %v23
  %v1311 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 %v23
  %v167 = and i32 %v20, 7
  %v1312 = or disjoint i32 %v167, 8
  %v1315 = or disjoint i32 %v167, 16
  %v1318 = or disjoint i32 %v167, 24
  %v1327 = icmp eq i64 %v23, 0
  br label %bb16

bb16:                                             ; preds = %bb16.lr.ph, %bb391
  %v10774 = phi i32 [ %v105, %bb16.lr.ph ], [ %v756, %bb391 ]
  %v10673 = phi float [ 0.000000e+00, %bb16.lr.ph ], [ %v754, %bb391 ]
  %v112 = sext i32 %v10774 to i64
  %v113 = getelementptr inbounds [4 x i8], ptr %v4, i64 %v112
  %v114 = load i32, ptr %v113, align 4
  %v115 = sext i32 %v114 to i64
  %6 = shl nsw i64 %v115, 13
  %v121 = add nsw i64 %6, %5
  br i1 %v12668, label %bb27, label %bb18.lr.ph

bb18.lr.ph:                                       ; preds = %bb16
  %gep71 = getelementptr i8, ptr %invariant.gep70, i64 %v121
  %v144 = shl i32 %v10774, 5
  br label %bb18

bb18:                                             ; preds = %bb18.lr.ph, %bb26
  %v12569 = phi i64 [ %v124, %bb18.lr.ph ], [ %v154, %bb26 ]
  %v128 = shl nuw nsw i64 %v12569, 8
  %gep = getelementptr i8, ptr %gep71, i64 %v128
  %v132 = load i8, ptr %gep, align 1
  %v133 = zext i8 %v132 to i32
  %v138 = lshr i32 %v133, 3
  %v141 = and i32 %v138, 15
  %v142 = and i32 %v133, 7
  %v143 = icmp eq i32 %v141, 0
  br i1 %v143, label %bb81, label %bb85

bb19:                                             ; preds = %bb176
  %v1282 = fadd float %v1281, %v1280
  %v145 = trunc nuw nsw i64 %v12569 to i32
  %v146 = add nuw nsw i32 %v144, %v145
  %v147.not = icmp slt i32 %v146, %v49
  %v149.not = icmp sgt i32 %v146, %v33
  %or.cond1 = select i1 %v147.not, i1 true, i1 %v149.not
  %v151 = fmul float %v1282, 6.250000e-02
  %v152 = select i1 %or.cond1, float -1.000000e+30, float %v151
  %v153 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @__shared_mem_23b72fdefd237786_1, i64 %v12569
  store float %v152, ptr addrspace(3) %v153, align 4
  br label %bb26

bb26:                                             ; preds = %bb176, %bb19
  %v154 = add nuw nsw i64 %v12569, 16
  %v126 = icmp samesign ugt i64 %v12569, 15
  br i1 %v126, label %bb27, label %bb18

bb27:                                             ; preds = %bb26, %bb16
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #8
  br i1 %v156, label %bb29, label %bb42

bb29:                                             ; preds = %bb27
  %v158 = load float, ptr addrspace(3) %v157, align 4
  %v159 = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v158, i32 1, i32 31) #8
  %v1284 = fcmp uno float %v158, 0.000000e+00
  %v1285 = fcmp oge float %v159, %v158
  %7 = select i1 %v1284, i1 true, i1 %v1285
  %v1287 = select i1 %7, float %v159, float %v158
  %v1288 = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1287, i32 2, i32 31) #8
  %v1289 = fcmp uno float %v1287, 0.000000e+00
  %v1290 = fcmp oge float %v1288, %v1287
  %8 = select i1 %v1289, i1 true, i1 %v1290
  %v1292 = select i1 %8, float %v1288, float %v1287
  %v1293 = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1292, i32 4, i32 31) #8
  %v1294 = fcmp uno float %v1292, 0.000000e+00
  %v1295 = fcmp oge float %v1293, %v1292
  %9 = select i1 %v1294, i1 true, i1 %v1295
  %v1297 = select i1 %9, float %v1293, float %v1292
  %v1298 = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1297, i32 8, i32 31) #8
  %v1299 = fcmp uno float %v1297, 0.000000e+00
  %v1300 = fcmp oge float %v1298, %v1297
  %10 = select i1 %v1299, i1 true, i1 %v1300
  %v1302 = select i1 %10, float %v1298, float %v1297
  %v1303 = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1302, i32 16, i32 31) #8
  %v1304 = fcmp uno float %v1302, 0.000000e+00
  %v1305 = fcmp oge float %v1303, %v1302
  %11 = select i1 %v1304, i1 true, i1 %v1305
  %v1307 = select i1 %11, float %v1303, float %v1302
  %v160 = load float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_0, align 4
  %v161.inv = fcmp ogt float %v1307, %v160
  %v163 = select i1 %v161.inv, float %v1307, float %v160
  %12 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #9
  %.not.i = icmp eq i32 %12, 0
  %v1309 = fsub float %v158, %v163
  %13 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v1309, float 5.724980e-03, float 5.000000e-01) #9
  %14 = tail call float @llvm.fma.f32(float %v1309, float 5.724980e-03, float 5.000000e-01)
  %.02.i62 = select i1 %.not.i, float %14, float %13
  %15 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i62) #9
  %16 = tail call float @llvm.nvvm.saturate.f(float %.02.i62) #9
  %.03.i63 = select i1 %.not.i, float %16, float %15
  %17 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i63, float 2.520000e+02, float f0x4B400001) #9
  %18 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i63, float 2.520000e+02, float f0x4B400001) #9
  %.04.i64 = select i1 %.not.i, float %18, float %17
  %19 = fadd float %.04.i64, f0xCB40007F
  %20 = fneg float %19
  %21 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v1309, float f0x3FB8AA3B, float %20) #9
  %22 = tail call float @llvm.fma.f32(float %v1309, float f0x3FB8AA3B, float %20)
  %.0.i65 = select i1 %.not.i, float %22, float %21
  %23 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v1309, float f0x32A57060, float %.0.i65) #9
  %24 = tail call float @llvm.fma.f32(float %v1309, float f0x32A57060, float %.0.i65)
  %.01.i66 = select i1 %.not.i, float %24, float %23
  %25 = bitcast float %.04.i64 to i32
  %26 = shl i32 %25, 23
  %27 = bitcast i32 %26 to float
  %28 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i66)
  %29 = fmul float %28, %27
  store float %29, ptr addrspace(3) %v1311, align 4
  %v168 = tail call float @llvm.nvvm.shfl.sync.idx.f32(i32 -1, float %29, i32 %v167, i32 31) #8
  %v1313 = tail call float @llvm.nvvm.shfl.sync.idx.f32(i32 -1, float %29, i32 %v1312, i32 31) #8
  %v1314 = fadd float %v168, %v1313
  %v1316 = tail call float @llvm.nvvm.shfl.sync.idx.f32(i32 -1, float %29, i32 %v1315, i32 31) #8
  %v1317 = fadd float %v1314, %v1316
  %v1319 = tail call float @llvm.nvvm.shfl.sync.idx.f32(i32 -1, float %29, i32 %v1318, i32 31) #8
  %v1320 = fadd float %v1317, %v1319
  %v1321 = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1320, i32 1, i32 31) #8
  %v1322 = fadd float %v1320, %v1321
  %v1323 = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1322, i32 4, i32 31) #8
  %v1324 = fadd float %v1322, %v1323
  %v1325 = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1324, i32 2, i32 31) #8
  br i1 %v1327, label %bb36, label %bb42

bb36:                                             ; preds = %bb29
  %v164 = fsub float %v160, %v163
  %30 = tail call float @llvm.fma.f32(float %v164, float 5.724980e-03, float 5.000000e-01)
  %31 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v164, float 5.724980e-03, float 5.000000e-01) #9
  %.02.i = select i1 %.not.i, float %30, float %31
  %32 = tail call float @llvm.nvvm.saturate.f(float %.02.i) #9
  %33 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i) #9
  %.03.i = select i1 %.not.i, float %32, float %33
  %34 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i, float 2.520000e+02, float f0x4B400001) #9
  %35 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i, float 2.520000e+02, float f0x4B400001) #9
  %.04.i = select i1 %.not.i, float %34, float %35
  %36 = fadd float %.04.i, f0xCB40007F
  %37 = fneg float %36
  %38 = tail call float @llvm.fma.f32(float %v164, float f0x3FB8AA3B, float %37)
  %39 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v164, float f0x3FB8AA3B, float %37) #9
  %.0.i = select i1 %.not.i, float %38, float %39
  %40 = tail call float @llvm.fma.f32(float %v164, float f0x32A57060, float %.0.i)
  %41 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v164, float f0x32A57060, float %.0.i) #9
  %.01.i = select i1 %.not.i, float %40, float %41
  %42 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i)
  %43 = bitcast float %.04.i to i32
  %44 = shl i32 %43, 23
  %45 = bitcast i32 %44 to float
  %46 = fmul float %42, %45
  %v1326 = fadd float %v1324, %v1325
  store float %v163, ptr addrspace(3) @__shared_mem_23b72fdefd237786_0, align 4
  %v171 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_0, i64 4), align 4
  %v172 = fmul float %46, %v171
  %v174 = fadd float %v1326, %v172
  store float %v174, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_0, i64 4), align 4
  store float %46, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_0, i64 8), align 4
  br label %bb42

bb42:                                             ; preds = %bb36, %bb29, %bb27
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #8
  %47 = getelementptr i8, ptr %v2, i64 %v121
  %v178 = getelementptr i8, ptr %47, i64 %v21
  %v179 = load i8, ptr %v178, align 1
  %v180 = zext i8 %v179 to i32
  %v185 = lshr i32 %v180, 3
  %v188 = and i32 %v185, 15
  %v189 = and i32 %v180, 7
  %v190 = icmp eq i32 %v188, 0
  br i1 %v190, label %bb200, label %bb204

bb78:                                             ; preds = %bb391, %bb13
  %v106.lcssa = phi float [ 0.000000e+00, %bb13 ], [ %v754, %bb391 ]
  %v758 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_0, i64 4), align 4
  %48 = getelementptr inbounds nuw [4 x i8], ptr %v5, i64 %v51
  %v760 = getelementptr inbounds nuw [4 x i8], ptr %48, i64 %v21
  %v761 = fdiv float %v106.lcssa, %v758
  store float %v761, ptr %v760, align 4
  br label %bb80

bb80:                                             ; preds = %bb78, %bb4
  ret void

bb81:                                             ; preds = %bb18
  %v762 = uitofp nneg i32 %v142 to float
  %v763 = fmul nnan float %v762, f0x3B000000
  %v765 = fneg float %v763
  %v76413 = icmp slt i8 %v132, 0
  %v766 = select i1 %v76413, float %v765, float %v763
  br label %bb86

bb85:                                             ; preds = %bb18
  %v132.signext = sext i8 %v132 to i32
  %v137 = and i32 %v132.signext, -2147483648
  %v767 = shl nuw nsw i32 %v141, 23
  %v770 = add nuw nsw i32 %v767, 1006632960
  %v774 = shl nuw nsw i32 %v142, 20
  %49 = or disjoint i32 %v770, %v774
  %v775 = or disjoint i32 %49, %v137
  %v776 = bitcast i32 %v775 to float
  br label %bb86

bb86:                                             ; preds = %bb85, %bb81
  %v777 = phi float [ %v766, %bb81 ], [ %v776, %bb85 ]
  %v778 = fmul float %v8, %v777
  %v779 = fmul float %v55, %v778
  %v781 = getelementptr i8, ptr %gep, i64 1
  %v782 = load i8, ptr %v781, align 1
  %v783 = zext i8 %v782 to i32
  %v788 = lshr i32 %v783, 3
  %v791 = and i32 %v788, 15
  %v792 = and i32 %v783, 7
  %v793 = icmp eq i32 %v791, 0
  br i1 %v793, label %bb87, label %bb91

bb87:                                             ; preds = %bb86
  %v794 = uitofp nneg i32 %v792 to float
  %v795 = fmul nnan float %v794, f0x3B000000
  %v797 = fneg float %v795
  %v79614 = icmp slt i8 %v782, 0
  %v798 = select i1 %v79614, float %v797, float %v795
  br label %bb92

bb91:                                             ; preds = %bb86
  %v782.signext = sext i8 %v782 to i32
  %v787 = and i32 %v782.signext, -2147483648
  %v799 = shl nuw nsw i32 %v791, 23
  %v802 = add nuw nsw i32 %v799, 1006632960
  %v806 = shl nuw nsw i32 %v792, 20
  %50 = or disjoint i32 %v802, %v806
  %v807 = or disjoint i32 %50, %v787
  %v808 = bitcast i32 %v807 to float
  br label %bb92

bb92:                                             ; preds = %bb91, %bb87
  %v809 = phi float [ %v798, %bb87 ], [ %v808, %bb91 ]
  %v810 = fmul float %v8, %v809
  %v811 = fmul float %v58, %v810
  %v813 = getelementptr i8, ptr %gep, i64 2
  %v814 = load i8, ptr %v813, align 1
  %v815 = zext i8 %v814 to i32
  %v820 = lshr i32 %v815, 3
  %v823 = and i32 %v820, 15
  %v824 = and i32 %v815, 7
  %v825 = icmp eq i32 %v823, 0
  br i1 %v825, label %bb93, label %bb97

bb93:                                             ; preds = %bb92
  %v826 = uitofp nneg i32 %v824 to float
  %v827 = fmul nnan float %v826, f0x3B000000
  %v829 = fneg float %v827
  %v82815 = icmp slt i8 %v814, 0
  %v830 = select i1 %v82815, float %v829, float %v827
  br label %bb98

bb97:                                             ; preds = %bb92
  %v814.signext = sext i8 %v814 to i32
  %v819 = and i32 %v814.signext, -2147483648
  %v831 = shl nuw nsw i32 %v823, 23
  %v834 = add nuw nsw i32 %v831, 1006632960
  %v838 = shl nuw nsw i32 %v824, 20
  %51 = or disjoint i32 %v834, %v838
  %v839 = or disjoint i32 %51, %v819
  %v840 = bitcast i32 %v839 to float
  br label %bb98

bb98:                                             ; preds = %bb97, %bb93
  %v841 = phi float [ %v830, %bb93 ], [ %v840, %bb97 ]
  %v842 = fmul float %v8, %v841
  %v843 = fmul float %v61, %v842
  %v845 = getelementptr i8, ptr %gep, i64 3
  %v846 = load i8, ptr %v845, align 1
  %v847 = zext i8 %v846 to i32
  %v852 = lshr i32 %v847, 3
  %v855 = and i32 %v852, 15
  %v856 = and i32 %v847, 7
  %v857 = icmp eq i32 %v855, 0
  br i1 %v857, label %bb99, label %bb103

bb99:                                             ; preds = %bb98
  %v858 = uitofp nneg i32 %v856 to float
  %v859 = fmul nnan float %v858, f0x3B000000
  %v861 = fneg float %v859
  %v86016 = icmp slt i8 %v846, 0
  %v862 = select i1 %v86016, float %v861, float %v859
  br label %bb104

bb103:                                            ; preds = %bb98
  %v846.signext = sext i8 %v846 to i32
  %v851 = and i32 %v846.signext, -2147483648
  %v863 = shl nuw nsw i32 %v855, 23
  %v866 = add nuw nsw i32 %v863, 1006632960
  %v870 = shl nuw nsw i32 %v856, 20
  %52 = or disjoint i32 %v866, %v870
  %v871 = or disjoint i32 %52, %v851
  %v872 = bitcast i32 %v871 to float
  br label %bb104

bb104:                                            ; preds = %bb103, %bb99
  %v873 = phi float [ %v862, %bb99 ], [ %v872, %bb103 ]
  %v874 = fmul float %v8, %v873
  %v875 = fmul float %v64, %v874
  %v877 = getelementptr i8, ptr %gep, i64 4
  %v878 = load i8, ptr %v877, align 1
  %v879 = zext i8 %v878 to i32
  %v884 = lshr i32 %v879, 3
  %v887 = and i32 %v884, 15
  %v888 = and i32 %v879, 7
  %v889 = icmp eq i32 %v887, 0
  br i1 %v889, label %bb105, label %bb109

bb105:                                            ; preds = %bb104
  %v890 = uitofp nneg i32 %v888 to float
  %v891 = fmul nnan float %v890, f0x3B000000
  %v893 = fneg float %v891
  %v89217 = icmp slt i8 %v878, 0
  %v894 = select i1 %v89217, float %v893, float %v891
  br label %bb110

bb109:                                            ; preds = %bb104
  %v878.signext = sext i8 %v878 to i32
  %v883 = and i32 %v878.signext, -2147483648
  %v895 = shl nuw nsw i32 %v887, 23
  %v898 = add nuw nsw i32 %v895, 1006632960
  %v902 = shl nuw nsw i32 %v888, 20
  %53 = or disjoint i32 %v898, %v902
  %v903 = or disjoint i32 %53, %v883
  %v904 = bitcast i32 %v903 to float
  br label %bb110

bb110:                                            ; preds = %bb109, %bb105
  %v905 = phi float [ %v894, %bb105 ], [ %v904, %bb109 ]
  %v906 = fmul float %v8, %v905
  %v907 = fmul float %v67, %v906
  %v909 = getelementptr i8, ptr %gep, i64 5
  %v910 = load i8, ptr %v909, align 1
  %v911 = zext i8 %v910 to i32
  %v916 = lshr i32 %v911, 3
  %v919 = and i32 %v916, 15
  %v920 = and i32 %v911, 7
  %v921 = icmp eq i32 %v919, 0
  br i1 %v921, label %bb111, label %bb115

bb111:                                            ; preds = %bb110
  %v922 = uitofp nneg i32 %v920 to float
  %v923 = fmul nnan float %v922, f0x3B000000
  %v925 = fneg float %v923
  %v92418 = icmp slt i8 %v910, 0
  %v926 = select i1 %v92418, float %v925, float %v923
  br label %bb116

bb115:                                            ; preds = %bb110
  %v910.signext = sext i8 %v910 to i32
  %v915 = and i32 %v910.signext, -2147483648
  %v927 = shl nuw nsw i32 %v919, 23
  %v930 = add nuw nsw i32 %v927, 1006632960
  %v934 = shl nuw nsw i32 %v920, 20
  %54 = or disjoint i32 %v930, %v934
  %v935 = or disjoint i32 %54, %v915
  %v936 = bitcast i32 %v935 to float
  br label %bb116

bb116:                                            ; preds = %bb115, %bb111
  %v937 = phi float [ %v926, %bb111 ], [ %v936, %bb115 ]
  %v938 = fmul float %v8, %v937
  %v939 = fmul float %v70, %v938
  %v941 = getelementptr i8, ptr %gep, i64 6
  %v942 = load i8, ptr %v941, align 1
  %v943 = zext i8 %v942 to i32
  %v948 = lshr i32 %v943, 3
  %v951 = and i32 %v948, 15
  %v952 = and i32 %v943, 7
  %v953 = icmp eq i32 %v951, 0
  br i1 %v953, label %bb117, label %bb121

bb117:                                            ; preds = %bb116
  %v954 = uitofp nneg i32 %v952 to float
  %v955 = fmul nnan float %v954, f0x3B000000
  %v957 = fneg float %v955
  %v95619 = icmp slt i8 %v942, 0
  %v958 = select i1 %v95619, float %v957, float %v955
  br label %bb122

bb121:                                            ; preds = %bb116
  %v942.signext = sext i8 %v942 to i32
  %v947 = and i32 %v942.signext, -2147483648
  %v959 = shl nuw nsw i32 %v951, 23
  %v962 = add nuw nsw i32 %v959, 1006632960
  %v966 = shl nuw nsw i32 %v952, 20
  %55 = or disjoint i32 %v962, %v966
  %v967 = or disjoint i32 %55, %v947
  %v968 = bitcast i32 %v967 to float
  br label %bb122

bb122:                                            ; preds = %bb121, %bb117
  %v969 = phi float [ %v958, %bb117 ], [ %v968, %bb121 ]
  %v970 = fmul float %v8, %v969
  %v971 = fmul float %v73, %v970
  %v973 = getelementptr i8, ptr %gep, i64 7
  %v974 = load i8, ptr %v973, align 1
  %v975 = zext i8 %v974 to i32
  %v980 = lshr i32 %v975, 3
  %v983 = and i32 %v980, 15
  %v984 = and i32 %v975, 7
  %v985 = icmp eq i32 %v983, 0
  br i1 %v985, label %bb123, label %bb127

bb123:                                            ; preds = %bb122
  %v986 = uitofp nneg i32 %v984 to float
  %v987 = fmul nnan float %v986, f0x3B000000
  %v989 = fneg float %v987
  %v98820 = icmp slt i8 %v974, 0
  %v990 = select i1 %v98820, float %v989, float %v987
  br label %bb128

bb127:                                            ; preds = %bb122
  %v974.signext = sext i8 %v974 to i32
  %v979 = and i32 %v974.signext, -2147483648
  %v991 = shl nuw nsw i32 %v983, 23
  %v994 = add nuw nsw i32 %v991, 1006632960
  %v998 = shl nuw nsw i32 %v984, 20
  %56 = or disjoint i32 %v994, %v998
  %v999 = or disjoint i32 %56, %v979
  %v1000 = bitcast i32 %v999 to float
  br label %bb128

bb128:                                            ; preds = %bb127, %bb123
  %v1001 = phi float [ %v990, %bb123 ], [ %v1000, %bb127 ]
  %v1002 = fmul float %v8, %v1001
  %v1003 = fmul float %v76, %v1002
  %v1005 = getelementptr i8, ptr %gep, i64 8
  %v1006 = load i8, ptr %v1005, align 1
  %v1007 = zext i8 %v1006 to i32
  %v1012 = lshr i32 %v1007, 3
  %v1015 = and i32 %v1012, 15
  %v1016 = and i32 %v1007, 7
  %v1017 = icmp eq i32 %v1015, 0
  br i1 %v1017, label %bb129, label %bb133

bb129:                                            ; preds = %bb128
  %v1018 = uitofp nneg i32 %v1016 to float
  %v1019 = fmul nnan float %v1018, f0x3B000000
  %v1021 = fneg float %v1019
  %v102021 = icmp slt i8 %v1006, 0
  %v1022 = select i1 %v102021, float %v1021, float %v1019
  br label %bb134

bb133:                                            ; preds = %bb128
  %v1006.signext = sext i8 %v1006 to i32
  %v1011 = and i32 %v1006.signext, -2147483648
  %v1023 = shl nuw nsw i32 %v1015, 23
  %v1026 = add nuw nsw i32 %v1023, 1006632960
  %v1030 = shl nuw nsw i32 %v1016, 20
  %57 = or disjoint i32 %v1026, %v1030
  %v1031 = or disjoint i32 %57, %v1011
  %v1032 = bitcast i32 %v1031 to float
  br label %bb134

bb134:                                            ; preds = %bb133, %bb129
  %v1033 = phi float [ %v1022, %bb129 ], [ %v1032, %bb133 ]
  %v1034 = fmul float %v8, %v1033
  %v1035 = fmul float %v79, %v1034
  %v1037 = getelementptr i8, ptr %gep, i64 9
  %v1038 = load i8, ptr %v1037, align 1
  %v1039 = zext i8 %v1038 to i32
  %v1044 = lshr i32 %v1039, 3
  %v1047 = and i32 %v1044, 15
  %v1048 = and i32 %v1039, 7
  %v1049 = icmp eq i32 %v1047, 0
  br i1 %v1049, label %bb135, label %bb139

bb135:                                            ; preds = %bb134
  %v1050 = uitofp nneg i32 %v1048 to float
  %v1051 = fmul nnan float %v1050, f0x3B000000
  %v1053 = fneg float %v1051
  %v105222 = icmp slt i8 %v1038, 0
  %v1054 = select i1 %v105222, float %v1053, float %v1051
  br label %bb140

bb139:                                            ; preds = %bb134
  %v1038.signext = sext i8 %v1038 to i32
  %v1043 = and i32 %v1038.signext, -2147483648
  %v1055 = shl nuw nsw i32 %v1047, 23
  %v1058 = add nuw nsw i32 %v1055, 1006632960
  %v1062 = shl nuw nsw i32 %v1048, 20
  %58 = or disjoint i32 %v1058, %v1062
  %v1063 = or disjoint i32 %58, %v1043
  %v1064 = bitcast i32 %v1063 to float
  br label %bb140

bb140:                                            ; preds = %bb139, %bb135
  %v1065 = phi float [ %v1054, %bb135 ], [ %v1064, %bb139 ]
  %v1066 = fmul float %v8, %v1065
  %v1067 = fmul float %v82, %v1066
  %v1069 = getelementptr i8, ptr %gep, i64 10
  %v1070 = load i8, ptr %v1069, align 1
  %v1071 = zext i8 %v1070 to i32
  %v1076 = lshr i32 %v1071, 3
  %v1079 = and i32 %v1076, 15
  %v1080 = and i32 %v1071, 7
  %v1081 = icmp eq i32 %v1079, 0
  br i1 %v1081, label %bb141, label %bb145

bb141:                                            ; preds = %bb140
  %v1082 = uitofp nneg i32 %v1080 to float
  %v1083 = fmul nnan float %v1082, f0x3B000000
  %v1085 = fneg float %v1083
  %v108423 = icmp slt i8 %v1070, 0
  %v1086 = select i1 %v108423, float %v1085, float %v1083
  br label %bb146

bb145:                                            ; preds = %bb140
  %v1070.signext = sext i8 %v1070 to i32
  %v1075 = and i32 %v1070.signext, -2147483648
  %v1087 = shl nuw nsw i32 %v1079, 23
  %v1090 = add nuw nsw i32 %v1087, 1006632960
  %v1094 = shl nuw nsw i32 %v1080, 20
  %59 = or disjoint i32 %v1090, %v1094
  %v1095 = or disjoint i32 %59, %v1075
  %v1096 = bitcast i32 %v1095 to float
  br label %bb146

bb146:                                            ; preds = %bb145, %bb141
  %v1097 = phi float [ %v1086, %bb141 ], [ %v1096, %bb145 ]
  %v1098 = fmul float %v8, %v1097
  %v1099 = fmul float %v85, %v1098
  %v1101 = getelementptr i8, ptr %gep, i64 11
  %v1102 = load i8, ptr %v1101, align 1
  %v1103 = zext i8 %v1102 to i32
  %v1108 = lshr i32 %v1103, 3
  %v1111 = and i32 %v1108, 15
  %v1112 = and i32 %v1103, 7
  %v1113 = icmp eq i32 %v1111, 0
  br i1 %v1113, label %bb147, label %bb151

bb147:                                            ; preds = %bb146
  %v1114 = uitofp nneg i32 %v1112 to float
  %v1115 = fmul nnan float %v1114, f0x3B000000
  %v1117 = fneg float %v1115
  %v111624 = icmp slt i8 %v1102, 0
  %v1118 = select i1 %v111624, float %v1117, float %v1115
  br label %bb152

bb151:                                            ; preds = %bb146
  %v1102.signext = sext i8 %v1102 to i32
  %v1107 = and i32 %v1102.signext, -2147483648
  %v1119 = shl nuw nsw i32 %v1111, 23
  %v1122 = add nuw nsw i32 %v1119, 1006632960
  %v1126 = shl nuw nsw i32 %v1112, 20
  %60 = or disjoint i32 %v1122, %v1126
  %v1127 = or disjoint i32 %60, %v1107
  %v1128 = bitcast i32 %v1127 to float
  br label %bb152

bb152:                                            ; preds = %bb151, %bb147
  %v1129 = phi float [ %v1118, %bb147 ], [ %v1128, %bb151 ]
  %v1130 = fmul float %v8, %v1129
  %v1131 = fmul float %v88, %v1130
  %v1133 = getelementptr i8, ptr %gep, i64 12
  %v1134 = load i8, ptr %v1133, align 1
  %v1135 = zext i8 %v1134 to i32
  %v1140 = lshr i32 %v1135, 3
  %v1143 = and i32 %v1140, 15
  %v1144 = and i32 %v1135, 7
  %v1145 = icmp eq i32 %v1143, 0
  br i1 %v1145, label %bb153, label %bb157

bb153:                                            ; preds = %bb152
  %v1146 = uitofp nneg i32 %v1144 to float
  %v1147 = fmul nnan float %v1146, f0x3B000000
  %v1149 = fneg float %v1147
  %v114825 = icmp slt i8 %v1134, 0
  %v1150 = select i1 %v114825, float %v1149, float %v1147
  br label %bb158

bb157:                                            ; preds = %bb152
  %v1134.signext = sext i8 %v1134 to i32
  %v1139 = and i32 %v1134.signext, -2147483648
  %v1151 = shl nuw nsw i32 %v1143, 23
  %v1154 = add nuw nsw i32 %v1151, 1006632960
  %v1158 = shl nuw nsw i32 %v1144, 20
  %61 = or disjoint i32 %v1154, %v1158
  %v1159 = or disjoint i32 %61, %v1139
  %v1160 = bitcast i32 %v1159 to float
  br label %bb158

bb158:                                            ; preds = %bb157, %bb153
  %v1161 = phi float [ %v1150, %bb153 ], [ %v1160, %bb157 ]
  %v1162 = fmul float %v8, %v1161
  %v1163 = fmul float %v91, %v1162
  %v1165 = getelementptr i8, ptr %gep, i64 13
  %v1166 = load i8, ptr %v1165, align 1
  %v1167 = zext i8 %v1166 to i32
  %v1172 = lshr i32 %v1167, 3
  %v1175 = and i32 %v1172, 15
  %v1176 = and i32 %v1167, 7
  %v1177 = icmp eq i32 %v1175, 0
  br i1 %v1177, label %bb159, label %bb163

bb159:                                            ; preds = %bb158
  %v1178 = uitofp nneg i32 %v1176 to float
  %v1179 = fmul nnan float %v1178, f0x3B000000
  %v1181 = fneg float %v1179
  %v118026 = icmp slt i8 %v1166, 0
  %v1182 = select i1 %v118026, float %v1181, float %v1179
  br label %bb164

bb163:                                            ; preds = %bb158
  %v1166.signext = sext i8 %v1166 to i32
  %v1171 = and i32 %v1166.signext, -2147483648
  %v1183 = shl nuw nsw i32 %v1175, 23
  %v1186 = add nuw nsw i32 %v1183, 1006632960
  %v1190 = shl nuw nsw i32 %v1176, 20
  %62 = or disjoint i32 %v1186, %v1190
  %v1191 = or disjoint i32 %62, %v1171
  %v1192 = bitcast i32 %v1191 to float
  br label %bb164

bb164:                                            ; preds = %bb163, %bb159
  %v1193 = phi float [ %v1182, %bb159 ], [ %v1192, %bb163 ]
  %v1194 = fmul float %v8, %v1193
  %v1195 = fmul float %v94, %v1194
  %v1197 = getelementptr i8, ptr %gep, i64 14
  %v1198 = load i8, ptr %v1197, align 1
  %v1199 = zext i8 %v1198 to i32
  %v1204 = lshr i32 %v1199, 3
  %v1207 = and i32 %v1204, 15
  %v1208 = and i32 %v1199, 7
  %v1209 = icmp eq i32 %v1207, 0
  br i1 %v1209, label %bb165, label %bb169

bb165:                                            ; preds = %bb164
  %v1210 = uitofp nneg i32 %v1208 to float
  %v1211 = fmul nnan float %v1210, f0x3B000000
  %v1213 = fneg float %v1211
  %v121227 = icmp slt i8 %v1198, 0
  %v1214 = select i1 %v121227, float %v1213, float %v1211
  br label %bb170

bb169:                                            ; preds = %bb164
  %v1198.signext = sext i8 %v1198 to i32
  %v1203 = and i32 %v1198.signext, -2147483648
  %v1215 = shl nuw nsw i32 %v1207, 23
  %v1218 = add nuw nsw i32 %v1215, 1006632960
  %v1222 = shl nuw nsw i32 %v1208, 20
  %63 = or disjoint i32 %v1218, %v1222
  %v1223 = or disjoint i32 %63, %v1203
  %v1224 = bitcast i32 %v1223 to float
  br label %bb170

bb170:                                            ; preds = %bb169, %bb165
  %v1225 = phi float [ %v1214, %bb165 ], [ %v1224, %bb169 ]
  %v1226 = fmul float %v8, %v1225
  %v1227 = fmul float %v97, %v1226
  %v1229 = getelementptr i8, ptr %gep, i64 15
  %v1230 = load i8, ptr %v1229, align 1
  %v1231 = zext i8 %v1230 to i32
  %v1236 = lshr i32 %v1231, 3
  %v1239 = and i32 %v1236, 15
  %v1240 = and i32 %v1231, 7
  %v1241 = icmp eq i32 %v1239, 0
  br i1 %v1241, label %bb171, label %bb175

bb171:                                            ; preds = %bb170
  %v1242 = uitofp nneg i32 %v1240 to float
  %v1243 = fmul nnan float %v1242, f0x3B000000
  %v1245 = fneg float %v1243
  %v124428 = icmp slt i8 %v1230, 0
  %v1246 = select i1 %v124428, float %v1245, float %v1243
  br label %bb176

bb175:                                            ; preds = %bb170
  %v1230.signext = sext i8 %v1230 to i32
  %v1235 = and i32 %v1230.signext, -2147483648
  %v1247 = shl nuw nsw i32 %v1239, 23
  %v1250 = add nuw nsw i32 %v1247, 1006632960
  %v1254 = shl nuw nsw i32 %v1240, 20
  %64 = or disjoint i32 %v1250, %v1254
  %v1255 = or disjoint i32 %64, %v1235
  %v1256 = bitcast i32 %v1255 to float
  br label %bb176

bb176:                                            ; preds = %bb175, %bb171
  %v1257 = phi float [ %v1246, %bb171 ], [ %v1256, %bb175 ]
  %v1258 = fmul float %v8, %v1257
  %v1259 = fmul float %v100, %v1258
  %v1260 = fadd float %v779, %v843
  %v1261 = fadd float %v907, %v971
  %v1262 = fadd float %v1260, %v1261
  %v1263 = fadd float %v1035, %v1099
  %v1264 = fadd float %v1163, %v1227
  %v1265 = fadd float %v1263, %v1264
  %v1266 = fadd float %v1262, %v1265
  %v1267 = fadd float %v811, %v875
  %v1268 = fadd float %v939, %v1003
  %v1269 = fadd float %v1267, %v1268
  %v1270 = fadd float %v1067, %v1131
  %v1271 = fadd float %v1195, %v1259
  %v1272 = fadd float %v1270, %v1271
  %v1273 = fadd float %v1269, %v1272
  %v1274 = fadd float %v1266, %v1273
  %v1275 = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1274, i32 8, i32 31) #8
  %v1276 = fadd float %v1275, %v1274
  %v1277 = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1276, i32 4, i32 31) #8
  %v1278 = fadd float %v1277, %v1276
  %v1279 = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1278, i32 2, i32 31) #8
  %v1280 = fadd float %v1279, %v1278
  %v1281 = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1280, i32 1, i32 31) #8
  br i1 %v1283, label %bb19, label %bb26

bb200:                                            ; preds = %bb42
  %v1328 = uitofp nneg i32 %v189 to float
  %v1329 = fmul nnan float %v1328, f0x3B000000
  %v1331 = fneg float %v1329
  %v133029 = icmp slt i8 %v179, 0
  %v1332 = select i1 %v133029, float %v1331, float %v1329
  br label %bb205

bb204:                                            ; preds = %bb42
  %v179.signext = sext i8 %v179 to i32
  %v184 = and i32 %v179.signext, -2147483648
  %v1333 = shl nuw nsw i32 %v188, 23
  %v1336 = add nuw nsw i32 %v1333, 1006632960
  %v1340 = shl nuw nsw i32 %v189, 20
  %65 = or disjoint i32 %v1336, %v1340
  %v1341 = or disjoint i32 %65, %v184
  %v1342 = bitcast i32 %v1341 to float
  br label %bb205

bb205:                                            ; preds = %bb204, %bb200
  %v1343 = phi float [ %v1332, %bb200 ], [ %v1342, %bb204 ]
  %v1344 = fmul float %v9, %v1343
  %v191 = load float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, align 4
  %v192 = fmul float %v1344, %v191
  %66 = getelementptr i8, ptr %47, i64 256
  %v195 = getelementptr i8, ptr %66, i64 %v21
  %v196 = load i8, ptr %v195, align 1
  %v197 = zext i8 %v196 to i32
  %v202 = lshr i32 %v197, 3
  %v205 = and i32 %v202, 15
  %v206 = and i32 %v197, 7
  %v207 = icmp eq i32 %v205, 0
  br i1 %v207, label %bb206, label %bb210

bb206:                                            ; preds = %bb205
  %v1346 = uitofp nneg i32 %v206 to float
  %v1347 = fmul nnan float %v1346, f0x3B000000
  %v1349 = fneg float %v1347
  %v134830 = icmp slt i8 %v196, 0
  %v1350 = select i1 %v134830, float %v1349, float %v1347
  br label %bb211

bb210:                                            ; preds = %bb205
  %v196.signext = sext i8 %v196 to i32
  %v201 = and i32 %v196.signext, -2147483648
  %v1351 = shl nuw nsw i32 %v205, 23
  %v1354 = add nuw nsw i32 %v1351, 1006632960
  %v1358 = shl nuw nsw i32 %v206, 20
  %67 = or disjoint i32 %v1354, %v1358
  %v1359 = or disjoint i32 %67, %v201
  %v1360 = bitcast i32 %v1359 to float
  br label %bb211

bb211:                                            ; preds = %bb210, %bb206
  %v1361 = phi float [ %v1350, %bb206 ], [ %v1360, %bb210 ]
  %v1362 = fmul float %v9, %v1361
  %v208 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 4), align 4
  %v209 = fmul float %v1362, %v208
  %68 = getelementptr i8, ptr %47, i64 512
  %v212 = getelementptr i8, ptr %68, i64 %v21
  %v213 = load i8, ptr %v212, align 1
  %v214 = zext i8 %v213 to i32
  %v219 = lshr i32 %v214, 3
  %v222 = and i32 %v219, 15
  %v223 = and i32 %v214, 7
  %v224 = icmp eq i32 %v222, 0
  br i1 %v224, label %bb212, label %bb216

bb212:                                            ; preds = %bb211
  %v1364 = uitofp nneg i32 %v223 to float
  %v1365 = fmul nnan float %v1364, f0x3B000000
  %v1367 = fneg float %v1365
  %v136631 = icmp slt i8 %v213, 0
  %v1368 = select i1 %v136631, float %v1367, float %v1365
  br label %bb217

bb216:                                            ; preds = %bb211
  %v213.signext = sext i8 %v213 to i32
  %v218 = and i32 %v213.signext, -2147483648
  %v1369 = shl nuw nsw i32 %v222, 23
  %v1372 = add nuw nsw i32 %v1369, 1006632960
  %v1376 = shl nuw nsw i32 %v223, 20
  %69 = or disjoint i32 %v1372, %v1376
  %v1377 = or disjoint i32 %69, %v218
  %v1378 = bitcast i32 %v1377 to float
  br label %bb217

bb217:                                            ; preds = %bb216, %bb212
  %v1379 = phi float [ %v1368, %bb212 ], [ %v1378, %bb216 ]
  %v1380 = fmul float %v9, %v1379
  %v225 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 8), align 4
  %v226 = fmul float %v1380, %v225
  %70 = getelementptr i8, ptr %47, i64 768
  %v229 = getelementptr i8, ptr %70, i64 %v21
  %v230 = load i8, ptr %v229, align 1
  %v231 = zext i8 %v230 to i32
  %v236 = lshr i32 %v231, 3
  %v239 = and i32 %v236, 15
  %v240 = and i32 %v231, 7
  %v241 = icmp eq i32 %v239, 0
  br i1 %v241, label %bb218, label %bb222

bb218:                                            ; preds = %bb217
  %v1382 = uitofp nneg i32 %v240 to float
  %v1383 = fmul nnan float %v1382, f0x3B000000
  %v1385 = fneg float %v1383
  %v138432 = icmp slt i8 %v230, 0
  %v1386 = select i1 %v138432, float %v1385, float %v1383
  br label %bb223

bb222:                                            ; preds = %bb217
  %v230.signext = sext i8 %v230 to i32
  %v235 = and i32 %v230.signext, -2147483648
  %v1387 = shl nuw nsw i32 %v239, 23
  %v1390 = add nuw nsw i32 %v1387, 1006632960
  %v1394 = shl nuw nsw i32 %v240, 20
  %71 = or disjoint i32 %v1390, %v1394
  %v1395 = or disjoint i32 %71, %v235
  %v1396 = bitcast i32 %v1395 to float
  br label %bb223

bb223:                                            ; preds = %bb222, %bb218
  %v1397 = phi float [ %v1386, %bb218 ], [ %v1396, %bb222 ]
  %v1398 = fmul float %v9, %v1397
  %v242 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 12), align 4
  %v243 = fmul float %v1398, %v242
  %72 = getelementptr i8, ptr %47, i64 1024
  %v246 = getelementptr i8, ptr %72, i64 %v21
  %v247 = load i8, ptr %v246, align 1
  %v248 = zext i8 %v247 to i32
  %v253 = lshr i32 %v248, 3
  %v256 = and i32 %v253, 15
  %v257 = and i32 %v248, 7
  %v258 = icmp eq i32 %v256, 0
  br i1 %v258, label %bb224, label %bb228

bb224:                                            ; preds = %bb223
  %v1400 = uitofp nneg i32 %v257 to float
  %v1401 = fmul nnan float %v1400, f0x3B000000
  %v1403 = fneg float %v1401
  %v140233 = icmp slt i8 %v247, 0
  %v1404 = select i1 %v140233, float %v1403, float %v1401
  br label %bb229

bb228:                                            ; preds = %bb223
  %v247.signext = sext i8 %v247 to i32
  %v252 = and i32 %v247.signext, -2147483648
  %v1405 = shl nuw nsw i32 %v256, 23
  %v1408 = add nuw nsw i32 %v1405, 1006632960
  %v1412 = shl nuw nsw i32 %v257, 20
  %73 = or disjoint i32 %v1408, %v1412
  %v1413 = or disjoint i32 %73, %v252
  %v1414 = bitcast i32 %v1413 to float
  br label %bb229

bb229:                                            ; preds = %bb228, %bb224
  %v1415 = phi float [ %v1404, %bb224 ], [ %v1414, %bb228 ]
  %v1416 = fmul float %v9, %v1415
  %v259 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 16), align 4
  %v260 = fmul float %v1416, %v259
  %74 = getelementptr i8, ptr %47, i64 1280
  %v263 = getelementptr i8, ptr %74, i64 %v21
  %v264 = load i8, ptr %v263, align 1
  %v265 = zext i8 %v264 to i32
  %v270 = lshr i32 %v265, 3
  %v273 = and i32 %v270, 15
  %v274 = and i32 %v265, 7
  %v275 = icmp eq i32 %v273, 0
  br i1 %v275, label %bb230, label %bb234

bb230:                                            ; preds = %bb229
  %v1418 = uitofp nneg i32 %v274 to float
  %v1419 = fmul nnan float %v1418, f0x3B000000
  %v1421 = fneg float %v1419
  %v142034 = icmp slt i8 %v264, 0
  %v1422 = select i1 %v142034, float %v1421, float %v1419
  br label %bb235

bb234:                                            ; preds = %bb229
  %v264.signext = sext i8 %v264 to i32
  %v269 = and i32 %v264.signext, -2147483648
  %v1423 = shl nuw nsw i32 %v273, 23
  %v1426 = add nuw nsw i32 %v1423, 1006632960
  %v1430 = shl nuw nsw i32 %v274, 20
  %75 = or disjoint i32 %v1426, %v1430
  %v1431 = or disjoint i32 %75, %v269
  %v1432 = bitcast i32 %v1431 to float
  br label %bb235

bb235:                                            ; preds = %bb234, %bb230
  %v1433 = phi float [ %v1422, %bb230 ], [ %v1432, %bb234 ]
  %v1434 = fmul float %v9, %v1433
  %v276 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 20), align 4
  %v277 = fmul float %v1434, %v276
  %76 = getelementptr i8, ptr %47, i64 1536
  %v280 = getelementptr i8, ptr %76, i64 %v21
  %v281 = load i8, ptr %v280, align 1
  %v282 = zext i8 %v281 to i32
  %v287 = lshr i32 %v282, 3
  %v290 = and i32 %v287, 15
  %v291 = and i32 %v282, 7
  %v292 = icmp eq i32 %v290, 0
  br i1 %v292, label %bb236, label %bb240

bb236:                                            ; preds = %bb235
  %v1436 = uitofp nneg i32 %v291 to float
  %v1437 = fmul nnan float %v1436, f0x3B000000
  %v1439 = fneg float %v1437
  %v143835 = icmp slt i8 %v281, 0
  %v1440 = select i1 %v143835, float %v1439, float %v1437
  br label %bb241

bb240:                                            ; preds = %bb235
  %v281.signext = sext i8 %v281 to i32
  %v286 = and i32 %v281.signext, -2147483648
  %v1441 = shl nuw nsw i32 %v290, 23
  %v1444 = add nuw nsw i32 %v1441, 1006632960
  %v1448 = shl nuw nsw i32 %v291, 20
  %77 = or disjoint i32 %v1444, %v1448
  %v1449 = or disjoint i32 %77, %v286
  %v1450 = bitcast i32 %v1449 to float
  br label %bb241

bb241:                                            ; preds = %bb240, %bb236
  %v1451 = phi float [ %v1440, %bb236 ], [ %v1450, %bb240 ]
  %v1452 = fmul float %v9, %v1451
  %v293 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 24), align 4
  %v294 = fmul float %v1452, %v293
  %78 = getelementptr i8, ptr %47, i64 1792
  %v297 = getelementptr i8, ptr %78, i64 %v21
  %v298 = load i8, ptr %v297, align 1
  %v299 = zext i8 %v298 to i32
  %v304 = lshr i32 %v299, 3
  %v307 = and i32 %v304, 15
  %v308 = and i32 %v299, 7
  %v309 = icmp eq i32 %v307, 0
  br i1 %v309, label %bb242, label %bb246

bb242:                                            ; preds = %bb241
  %v1454 = uitofp nneg i32 %v308 to float
  %v1455 = fmul nnan float %v1454, f0x3B000000
  %v1457 = fneg float %v1455
  %v145636 = icmp slt i8 %v298, 0
  %v1458 = select i1 %v145636, float %v1457, float %v1455
  br label %bb247

bb246:                                            ; preds = %bb241
  %v298.signext = sext i8 %v298 to i32
  %v303 = and i32 %v298.signext, -2147483648
  %v1459 = shl nuw nsw i32 %v307, 23
  %v1462 = add nuw nsw i32 %v1459, 1006632960
  %v1466 = shl nuw nsw i32 %v308, 20
  %79 = or disjoint i32 %v1462, %v1466
  %v1467 = or disjoint i32 %79, %v303
  %v1468 = bitcast i32 %v1467 to float
  br label %bb247

bb247:                                            ; preds = %bb246, %bb242
  %v1469 = phi float [ %v1458, %bb242 ], [ %v1468, %bb246 ]
  %v1470 = fmul float %v9, %v1469
  %v310 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 28), align 4
  %v311 = fmul float %v1470, %v310
  %80 = getelementptr i8, ptr %47, i64 2048
  %v314 = getelementptr i8, ptr %80, i64 %v21
  %v315 = load i8, ptr %v314, align 1
  %v316 = zext i8 %v315 to i32
  %v321 = lshr i32 %v316, 3
  %v324 = and i32 %v321, 15
  %v325 = and i32 %v316, 7
  %v326 = icmp eq i32 %v324, 0
  br i1 %v326, label %bb248, label %bb252

bb248:                                            ; preds = %bb247
  %v1472 = uitofp nneg i32 %v325 to float
  %v1473 = fmul nnan float %v1472, f0x3B000000
  %v1475 = fneg float %v1473
  %v147437 = icmp slt i8 %v315, 0
  %v1476 = select i1 %v147437, float %v1475, float %v1473
  br label %bb253

bb252:                                            ; preds = %bb247
  %v315.signext = sext i8 %v315 to i32
  %v320 = and i32 %v315.signext, -2147483648
  %v1477 = shl nuw nsw i32 %v324, 23
  %v1480 = add nuw nsw i32 %v1477, 1006632960
  %v1484 = shl nuw nsw i32 %v325, 20
  %81 = or disjoint i32 %v1480, %v1484
  %v1485 = or disjoint i32 %81, %v320
  %v1486 = bitcast i32 %v1485 to float
  br label %bb253

bb253:                                            ; preds = %bb252, %bb248
  %v1487 = phi float [ %v1476, %bb248 ], [ %v1486, %bb252 ]
  %v1488 = fmul float %v9, %v1487
  %v327 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 32), align 4
  %v328 = fmul float %v1488, %v327
  %82 = getelementptr i8, ptr %47, i64 2304
  %v331 = getelementptr i8, ptr %82, i64 %v21
  %v332 = load i8, ptr %v331, align 1
  %v333 = zext i8 %v332 to i32
  %v338 = lshr i32 %v333, 3
  %v341 = and i32 %v338, 15
  %v342 = and i32 %v333, 7
  %v343 = icmp eq i32 %v341, 0
  br i1 %v343, label %bb254, label %bb258

bb254:                                            ; preds = %bb253
  %v1490 = uitofp nneg i32 %v342 to float
  %v1491 = fmul nnan float %v1490, f0x3B000000
  %v1493 = fneg float %v1491
  %v149238 = icmp slt i8 %v332, 0
  %v1494 = select i1 %v149238, float %v1493, float %v1491
  br label %bb259

bb258:                                            ; preds = %bb253
  %v332.signext = sext i8 %v332 to i32
  %v337 = and i32 %v332.signext, -2147483648
  %v1495 = shl nuw nsw i32 %v341, 23
  %v1498 = add nuw nsw i32 %v1495, 1006632960
  %v1502 = shl nuw nsw i32 %v342, 20
  %83 = or disjoint i32 %v1498, %v1502
  %v1503 = or disjoint i32 %83, %v337
  %v1504 = bitcast i32 %v1503 to float
  br label %bb259

bb259:                                            ; preds = %bb258, %bb254
  %v1505 = phi float [ %v1494, %bb254 ], [ %v1504, %bb258 ]
  %v1506 = fmul float %v9, %v1505
  %v344 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 36), align 4
  %v345 = fmul float %v1506, %v344
  %84 = getelementptr i8, ptr %47, i64 2560
  %v348 = getelementptr i8, ptr %84, i64 %v21
  %v349 = load i8, ptr %v348, align 1
  %v350 = zext i8 %v349 to i32
  %v355 = lshr i32 %v350, 3
  %v358 = and i32 %v355, 15
  %v359 = and i32 %v350, 7
  %v360 = icmp eq i32 %v358, 0
  br i1 %v360, label %bb260, label %bb264

bb260:                                            ; preds = %bb259
  %v1508 = uitofp nneg i32 %v359 to float
  %v1509 = fmul nnan float %v1508, f0x3B000000
  %v1511 = fneg float %v1509
  %v151039 = icmp slt i8 %v349, 0
  %v1512 = select i1 %v151039, float %v1511, float %v1509
  br label %bb265

bb264:                                            ; preds = %bb259
  %v349.signext = sext i8 %v349 to i32
  %v354 = and i32 %v349.signext, -2147483648
  %v1513 = shl nuw nsw i32 %v358, 23
  %v1516 = add nuw nsw i32 %v1513, 1006632960
  %v1520 = shl nuw nsw i32 %v359, 20
  %85 = or disjoint i32 %v1516, %v1520
  %v1521 = or disjoint i32 %85, %v354
  %v1522 = bitcast i32 %v1521 to float
  br label %bb265

bb265:                                            ; preds = %bb264, %bb260
  %v1523 = phi float [ %v1512, %bb260 ], [ %v1522, %bb264 ]
  %v1524 = fmul float %v9, %v1523
  %v361 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 40), align 4
  %v362 = fmul float %v1524, %v361
  %86 = getelementptr i8, ptr %47, i64 2816
  %v365 = getelementptr i8, ptr %86, i64 %v21
  %v366 = load i8, ptr %v365, align 1
  %v367 = zext i8 %v366 to i32
  %v372 = lshr i32 %v367, 3
  %v375 = and i32 %v372, 15
  %v376 = and i32 %v367, 7
  %v377 = icmp eq i32 %v375, 0
  br i1 %v377, label %bb266, label %bb270

bb266:                                            ; preds = %bb265
  %v1526 = uitofp nneg i32 %v376 to float
  %v1527 = fmul nnan float %v1526, f0x3B000000
  %v1529 = fneg float %v1527
  %v152840 = icmp slt i8 %v366, 0
  %v1530 = select i1 %v152840, float %v1529, float %v1527
  br label %bb271

bb270:                                            ; preds = %bb265
  %v366.signext = sext i8 %v366 to i32
  %v371 = and i32 %v366.signext, -2147483648
  %v1531 = shl nuw nsw i32 %v375, 23
  %v1534 = add nuw nsw i32 %v1531, 1006632960
  %v1538 = shl nuw nsw i32 %v376, 20
  %87 = or disjoint i32 %v1534, %v1538
  %v1539 = or disjoint i32 %87, %v371
  %v1540 = bitcast i32 %v1539 to float
  br label %bb271

bb271:                                            ; preds = %bb270, %bb266
  %v1541 = phi float [ %v1530, %bb266 ], [ %v1540, %bb270 ]
  %v1542 = fmul float %v9, %v1541
  %v378 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 44), align 4
  %v379 = fmul float %v1542, %v378
  %88 = getelementptr i8, ptr %47, i64 3072
  %v382 = getelementptr i8, ptr %88, i64 %v21
  %v383 = load i8, ptr %v382, align 1
  %v384 = zext i8 %v383 to i32
  %v389 = lshr i32 %v384, 3
  %v392 = and i32 %v389, 15
  %v393 = and i32 %v384, 7
  %v394 = icmp eq i32 %v392, 0
  br i1 %v394, label %bb272, label %bb276

bb272:                                            ; preds = %bb271
  %v1544 = uitofp nneg i32 %v393 to float
  %v1545 = fmul nnan float %v1544, f0x3B000000
  %v1547 = fneg float %v1545
  %v154641 = icmp slt i8 %v383, 0
  %v1548 = select i1 %v154641, float %v1547, float %v1545
  br label %bb277

bb276:                                            ; preds = %bb271
  %v383.signext = sext i8 %v383 to i32
  %v388 = and i32 %v383.signext, -2147483648
  %v1549 = shl nuw nsw i32 %v392, 23
  %v1552 = add nuw nsw i32 %v1549, 1006632960
  %v1556 = shl nuw nsw i32 %v393, 20
  %89 = or disjoint i32 %v1552, %v1556
  %v1557 = or disjoint i32 %89, %v388
  %v1558 = bitcast i32 %v1557 to float
  br label %bb277

bb277:                                            ; preds = %bb276, %bb272
  %v1559 = phi float [ %v1548, %bb272 ], [ %v1558, %bb276 ]
  %v1560 = fmul float %v9, %v1559
  %v395 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 48), align 4
  %v396 = fmul float %v1560, %v395
  %90 = getelementptr i8, ptr %47, i64 3328
  %v399 = getelementptr i8, ptr %90, i64 %v21
  %v400 = load i8, ptr %v399, align 1
  %v401 = zext i8 %v400 to i32
  %v406 = lshr i32 %v401, 3
  %v409 = and i32 %v406, 15
  %v410 = and i32 %v401, 7
  %v411 = icmp eq i32 %v409, 0
  br i1 %v411, label %bb278, label %bb282

bb278:                                            ; preds = %bb277
  %v1562 = uitofp nneg i32 %v410 to float
  %v1563 = fmul nnan float %v1562, f0x3B000000
  %v1565 = fneg float %v1563
  %v156442 = icmp slt i8 %v400, 0
  %v1566 = select i1 %v156442, float %v1565, float %v1563
  br label %bb283

bb282:                                            ; preds = %bb277
  %v400.signext = sext i8 %v400 to i32
  %v405 = and i32 %v400.signext, -2147483648
  %v1567 = shl nuw nsw i32 %v409, 23
  %v1570 = add nuw nsw i32 %v1567, 1006632960
  %v1574 = shl nuw nsw i32 %v410, 20
  %91 = or disjoint i32 %v1570, %v1574
  %v1575 = or disjoint i32 %91, %v405
  %v1576 = bitcast i32 %v1575 to float
  br label %bb283

bb283:                                            ; preds = %bb282, %bb278
  %v1577 = phi float [ %v1566, %bb278 ], [ %v1576, %bb282 ]
  %v1578 = fmul float %v9, %v1577
  %v412 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 52), align 4
  %v413 = fmul float %v1578, %v412
  %92 = getelementptr i8, ptr %47, i64 3584
  %v416 = getelementptr i8, ptr %92, i64 %v21
  %v417 = load i8, ptr %v416, align 1
  %v418 = zext i8 %v417 to i32
  %v423 = lshr i32 %v418, 3
  %v426 = and i32 %v423, 15
  %v427 = and i32 %v418, 7
  %v428 = icmp eq i32 %v426, 0
  br i1 %v428, label %bb284, label %bb288

bb284:                                            ; preds = %bb283
  %v1580 = uitofp nneg i32 %v427 to float
  %v1581 = fmul nnan float %v1580, f0x3B000000
  %v1583 = fneg float %v1581
  %v158243 = icmp slt i8 %v417, 0
  %v1584 = select i1 %v158243, float %v1583, float %v1581
  br label %bb289

bb288:                                            ; preds = %bb283
  %v417.signext = sext i8 %v417 to i32
  %v422 = and i32 %v417.signext, -2147483648
  %v1585 = shl nuw nsw i32 %v426, 23
  %v1588 = add nuw nsw i32 %v1585, 1006632960
  %v1592 = shl nuw nsw i32 %v427, 20
  %93 = or disjoint i32 %v1588, %v1592
  %v1593 = or disjoint i32 %93, %v422
  %v1594 = bitcast i32 %v1593 to float
  br label %bb289

bb289:                                            ; preds = %bb288, %bb284
  %v1595 = phi float [ %v1584, %bb284 ], [ %v1594, %bb288 ]
  %v1596 = fmul float %v9, %v1595
  %v429 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 56), align 4
  %v430 = fmul float %v1596, %v429
  %94 = getelementptr i8, ptr %47, i64 3840
  %v433 = getelementptr i8, ptr %94, i64 %v21
  %v434 = load i8, ptr %v433, align 1
  %v435 = zext i8 %v434 to i32
  %v440 = lshr i32 %v435, 3
  %v443 = and i32 %v440, 15
  %v444 = and i32 %v435, 7
  %v445 = icmp eq i32 %v443, 0
  br i1 %v445, label %bb290, label %bb294

bb290:                                            ; preds = %bb289
  %v1598 = uitofp nneg i32 %v444 to float
  %v1599 = fmul nnan float %v1598, f0x3B000000
  %v1601 = fneg float %v1599
  %v160044 = icmp slt i8 %v434, 0
  %v1602 = select i1 %v160044, float %v1601, float %v1599
  br label %bb295

bb294:                                            ; preds = %bb289
  %v434.signext = sext i8 %v434 to i32
  %v439 = and i32 %v434.signext, -2147483648
  %v1603 = shl nuw nsw i32 %v443, 23
  %v1606 = add nuw nsw i32 %v1603, 1006632960
  %v1610 = shl nuw nsw i32 %v444, 20
  %95 = or disjoint i32 %v1606, %v1610
  %v1611 = or disjoint i32 %95, %v439
  %v1612 = bitcast i32 %v1611 to float
  br label %bb295

bb295:                                            ; preds = %bb294, %bb290
  %v1613 = phi float [ %v1602, %bb290 ], [ %v1612, %bb294 ]
  %v1614 = fmul float %v9, %v1613
  %v446 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 60), align 4
  %v447 = fmul float %v1614, %v446
  %96 = getelementptr i8, ptr %47, i64 4096
  %v450 = getelementptr i8, ptr %96, i64 %v21
  %v451 = load i8, ptr %v450, align 1
  %v452 = zext i8 %v451 to i32
  %v457 = lshr i32 %v452, 3
  %v460 = and i32 %v457, 15
  %v461 = and i32 %v452, 7
  %v462 = icmp eq i32 %v460, 0
  br i1 %v462, label %bb296, label %bb300

bb296:                                            ; preds = %bb295
  %v1616 = uitofp nneg i32 %v461 to float
  %v1617 = fmul nnan float %v1616, f0x3B000000
  %v1619 = fneg float %v1617
  %v161845 = icmp slt i8 %v451, 0
  %v1620 = select i1 %v161845, float %v1619, float %v1617
  br label %bb301

bb300:                                            ; preds = %bb295
  %v451.signext = sext i8 %v451 to i32
  %v456 = and i32 %v451.signext, -2147483648
  %v1621 = shl nuw nsw i32 %v460, 23
  %v1624 = add nuw nsw i32 %v1621, 1006632960
  %v1628 = shl nuw nsw i32 %v461, 20
  %97 = or disjoint i32 %v1624, %v1628
  %v1629 = or disjoint i32 %97, %v456
  %v1630 = bitcast i32 %v1629 to float
  br label %bb301

bb301:                                            ; preds = %bb300, %bb296
  %v1631 = phi float [ %v1620, %bb296 ], [ %v1630, %bb300 ]
  %v1632 = fmul float %v9, %v1631
  %v463 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 64), align 4
  %v464 = fmul float %v1632, %v463
  %98 = getelementptr i8, ptr %47, i64 4352
  %v467 = getelementptr i8, ptr %98, i64 %v21
  %v468 = load i8, ptr %v467, align 1
  %v469 = zext i8 %v468 to i32
  %v474 = lshr i32 %v469, 3
  %v477 = and i32 %v474, 15
  %v478 = and i32 %v469, 7
  %v479 = icmp eq i32 %v477, 0
  br i1 %v479, label %bb302, label %bb306

bb302:                                            ; preds = %bb301
  %v1634 = uitofp nneg i32 %v478 to float
  %v1635 = fmul nnan float %v1634, f0x3B000000
  %v1637 = fneg float %v1635
  %v163646 = icmp slt i8 %v468, 0
  %v1638 = select i1 %v163646, float %v1637, float %v1635
  br label %bb307

bb306:                                            ; preds = %bb301
  %v468.signext = sext i8 %v468 to i32
  %v473 = and i32 %v468.signext, -2147483648
  %v1639 = shl nuw nsw i32 %v477, 23
  %v1642 = add nuw nsw i32 %v1639, 1006632960
  %v1646 = shl nuw nsw i32 %v478, 20
  %99 = or disjoint i32 %v1642, %v1646
  %v1647 = or disjoint i32 %99, %v473
  %v1648 = bitcast i32 %v1647 to float
  br label %bb307

bb307:                                            ; preds = %bb306, %bb302
  %v1649 = phi float [ %v1638, %bb302 ], [ %v1648, %bb306 ]
  %v1650 = fmul float %v9, %v1649
  %v480 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 68), align 4
  %v481 = fmul float %v1650, %v480
  %100 = getelementptr i8, ptr %47, i64 4608
  %v484 = getelementptr i8, ptr %100, i64 %v21
  %v485 = load i8, ptr %v484, align 1
  %v486 = zext i8 %v485 to i32
  %v491 = lshr i32 %v486, 3
  %v494 = and i32 %v491, 15
  %v495 = and i32 %v486, 7
  %v496 = icmp eq i32 %v494, 0
  br i1 %v496, label %bb308, label %bb312

bb308:                                            ; preds = %bb307
  %v1652 = uitofp nneg i32 %v495 to float
  %v1653 = fmul nnan float %v1652, f0x3B000000
  %v1655 = fneg float %v1653
  %v165447 = icmp slt i8 %v485, 0
  %v1656 = select i1 %v165447, float %v1655, float %v1653
  br label %bb313

bb312:                                            ; preds = %bb307
  %v485.signext = sext i8 %v485 to i32
  %v490 = and i32 %v485.signext, -2147483648
  %v1657 = shl nuw nsw i32 %v494, 23
  %v1660 = add nuw nsw i32 %v1657, 1006632960
  %v1664 = shl nuw nsw i32 %v495, 20
  %101 = or disjoint i32 %v1660, %v1664
  %v1665 = or disjoint i32 %101, %v490
  %v1666 = bitcast i32 %v1665 to float
  br label %bb313

bb313:                                            ; preds = %bb312, %bb308
  %v1667 = phi float [ %v1656, %bb308 ], [ %v1666, %bb312 ]
  %v1668 = fmul float %v9, %v1667
  %v497 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 72), align 4
  %v498 = fmul float %v1668, %v497
  %102 = getelementptr i8, ptr %47, i64 4864
  %v501 = getelementptr i8, ptr %102, i64 %v21
  %v502 = load i8, ptr %v501, align 1
  %v503 = zext i8 %v502 to i32
  %v508 = lshr i32 %v503, 3
  %v511 = and i32 %v508, 15
  %v512 = and i32 %v503, 7
  %v513 = icmp eq i32 %v511, 0
  br i1 %v513, label %bb314, label %bb318

bb314:                                            ; preds = %bb313
  %v1670 = uitofp nneg i32 %v512 to float
  %v1671 = fmul nnan float %v1670, f0x3B000000
  %v1673 = fneg float %v1671
  %v167248 = icmp slt i8 %v502, 0
  %v1674 = select i1 %v167248, float %v1673, float %v1671
  br label %bb319

bb318:                                            ; preds = %bb313
  %v502.signext = sext i8 %v502 to i32
  %v507 = and i32 %v502.signext, -2147483648
  %v1675 = shl nuw nsw i32 %v511, 23
  %v1678 = add nuw nsw i32 %v1675, 1006632960
  %v1682 = shl nuw nsw i32 %v512, 20
  %103 = or disjoint i32 %v1678, %v1682
  %v1683 = or disjoint i32 %103, %v507
  %v1684 = bitcast i32 %v1683 to float
  br label %bb319

bb319:                                            ; preds = %bb318, %bb314
  %v1685 = phi float [ %v1674, %bb314 ], [ %v1684, %bb318 ]
  %v1686 = fmul float %v9, %v1685
  %v514 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 76), align 4
  %v515 = fmul float %v1686, %v514
  %104 = getelementptr i8, ptr %47, i64 5120
  %v518 = getelementptr i8, ptr %104, i64 %v21
  %v519 = load i8, ptr %v518, align 1
  %v520 = zext i8 %v519 to i32
  %v525 = lshr i32 %v520, 3
  %v528 = and i32 %v525, 15
  %v529 = and i32 %v520, 7
  %v530 = icmp eq i32 %v528, 0
  br i1 %v530, label %bb320, label %bb324

bb320:                                            ; preds = %bb319
  %v1688 = uitofp nneg i32 %v529 to float
  %v1689 = fmul nnan float %v1688, f0x3B000000
  %v1691 = fneg float %v1689
  %v169049 = icmp slt i8 %v519, 0
  %v1692 = select i1 %v169049, float %v1691, float %v1689
  br label %bb325

bb324:                                            ; preds = %bb319
  %v519.signext = sext i8 %v519 to i32
  %v524 = and i32 %v519.signext, -2147483648
  %v1693 = shl nuw nsw i32 %v528, 23
  %v1696 = add nuw nsw i32 %v1693, 1006632960
  %v1700 = shl nuw nsw i32 %v529, 20
  %105 = or disjoint i32 %v1696, %v1700
  %v1701 = or disjoint i32 %105, %v524
  %v1702 = bitcast i32 %v1701 to float
  br label %bb325

bb325:                                            ; preds = %bb324, %bb320
  %v1703 = phi float [ %v1692, %bb320 ], [ %v1702, %bb324 ]
  %v1704 = fmul float %v9, %v1703
  %v531 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 80), align 4
  %v532 = fmul float %v1704, %v531
  %106 = getelementptr i8, ptr %47, i64 5376
  %v535 = getelementptr i8, ptr %106, i64 %v21
  %v536 = load i8, ptr %v535, align 1
  %v537 = zext i8 %v536 to i32
  %v542 = lshr i32 %v537, 3
  %v545 = and i32 %v542, 15
  %v546 = and i32 %v537, 7
  %v547 = icmp eq i32 %v545, 0
  br i1 %v547, label %bb326, label %bb330

bb326:                                            ; preds = %bb325
  %v1706 = uitofp nneg i32 %v546 to float
  %v1707 = fmul nnan float %v1706, f0x3B000000
  %v1709 = fneg float %v1707
  %v170850 = icmp slt i8 %v536, 0
  %v1710 = select i1 %v170850, float %v1709, float %v1707
  br label %bb331

bb330:                                            ; preds = %bb325
  %v536.signext = sext i8 %v536 to i32
  %v541 = and i32 %v536.signext, -2147483648
  %v1711 = shl nuw nsw i32 %v545, 23
  %v1714 = add nuw nsw i32 %v1711, 1006632960
  %v1718 = shl nuw nsw i32 %v546, 20
  %107 = or disjoint i32 %v1714, %v1718
  %v1719 = or disjoint i32 %107, %v541
  %v1720 = bitcast i32 %v1719 to float
  br label %bb331

bb331:                                            ; preds = %bb330, %bb326
  %v1721 = phi float [ %v1710, %bb326 ], [ %v1720, %bb330 ]
  %v1722 = fmul float %v9, %v1721
  %v548 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 84), align 4
  %v549 = fmul float %v1722, %v548
  %108 = getelementptr i8, ptr %47, i64 5632
  %v552 = getelementptr i8, ptr %108, i64 %v21
  %v553 = load i8, ptr %v552, align 1
  %v554 = zext i8 %v553 to i32
  %v559 = lshr i32 %v554, 3
  %v562 = and i32 %v559, 15
  %v563 = and i32 %v554, 7
  %v564 = icmp eq i32 %v562, 0
  br i1 %v564, label %bb332, label %bb336

bb332:                                            ; preds = %bb331
  %v1724 = uitofp nneg i32 %v563 to float
  %v1725 = fmul nnan float %v1724, f0x3B000000
  %v1727 = fneg float %v1725
  %v172651 = icmp slt i8 %v553, 0
  %v1728 = select i1 %v172651, float %v1727, float %v1725
  br label %bb337

bb336:                                            ; preds = %bb331
  %v553.signext = sext i8 %v553 to i32
  %v558 = and i32 %v553.signext, -2147483648
  %v1729 = shl nuw nsw i32 %v562, 23
  %v1732 = add nuw nsw i32 %v1729, 1006632960
  %v1736 = shl nuw nsw i32 %v563, 20
  %109 = or disjoint i32 %v1732, %v1736
  %v1737 = or disjoint i32 %109, %v558
  %v1738 = bitcast i32 %v1737 to float
  br label %bb337

bb337:                                            ; preds = %bb336, %bb332
  %v1739 = phi float [ %v1728, %bb332 ], [ %v1738, %bb336 ]
  %v1740 = fmul float %v9, %v1739
  %v565 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 88), align 4
  %v566 = fmul float %v1740, %v565
  %110 = getelementptr i8, ptr %47, i64 5888
  %v569 = getelementptr i8, ptr %110, i64 %v21
  %v570 = load i8, ptr %v569, align 1
  %v571 = zext i8 %v570 to i32
  %v576 = lshr i32 %v571, 3
  %v579 = and i32 %v576, 15
  %v580 = and i32 %v571, 7
  %v581 = icmp eq i32 %v579, 0
  br i1 %v581, label %bb338, label %bb342

bb338:                                            ; preds = %bb337
  %v1742 = uitofp nneg i32 %v580 to float
  %v1743 = fmul nnan float %v1742, f0x3B000000
  %v1745 = fneg float %v1743
  %v174452 = icmp slt i8 %v570, 0
  %v1746 = select i1 %v174452, float %v1745, float %v1743
  br label %bb343

bb342:                                            ; preds = %bb337
  %v570.signext = sext i8 %v570 to i32
  %v575 = and i32 %v570.signext, -2147483648
  %v1747 = shl nuw nsw i32 %v579, 23
  %v1750 = add nuw nsw i32 %v1747, 1006632960
  %v1754 = shl nuw nsw i32 %v580, 20
  %111 = or disjoint i32 %v1750, %v1754
  %v1755 = or disjoint i32 %111, %v575
  %v1756 = bitcast i32 %v1755 to float
  br label %bb343

bb343:                                            ; preds = %bb342, %bb338
  %v1757 = phi float [ %v1746, %bb338 ], [ %v1756, %bb342 ]
  %v1758 = fmul float %v9, %v1757
  %v582 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 92), align 4
  %v583 = fmul float %v1758, %v582
  %112 = getelementptr i8, ptr %47, i64 6144
  %v586 = getelementptr i8, ptr %112, i64 %v21
  %v587 = load i8, ptr %v586, align 1
  %v588 = zext i8 %v587 to i32
  %v593 = lshr i32 %v588, 3
  %v596 = and i32 %v593, 15
  %v597 = and i32 %v588, 7
  %v598 = icmp eq i32 %v596, 0
  br i1 %v598, label %bb344, label %bb348

bb344:                                            ; preds = %bb343
  %v1760 = uitofp nneg i32 %v597 to float
  %v1761 = fmul nnan float %v1760, f0x3B000000
  %v1763 = fneg float %v1761
  %v176253 = icmp slt i8 %v587, 0
  %v1764 = select i1 %v176253, float %v1763, float %v1761
  br label %bb349

bb348:                                            ; preds = %bb343
  %v587.signext = sext i8 %v587 to i32
  %v592 = and i32 %v587.signext, -2147483648
  %v1765 = shl nuw nsw i32 %v596, 23
  %v1768 = add nuw nsw i32 %v1765, 1006632960
  %v1772 = shl nuw nsw i32 %v597, 20
  %113 = or disjoint i32 %v1768, %v1772
  %v1773 = or disjoint i32 %113, %v592
  %v1774 = bitcast i32 %v1773 to float
  br label %bb349

bb349:                                            ; preds = %bb348, %bb344
  %v1775 = phi float [ %v1764, %bb344 ], [ %v1774, %bb348 ]
  %v1776 = fmul float %v9, %v1775
  %v599 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 96), align 4
  %v600 = fmul float %v1776, %v599
  %114 = getelementptr i8, ptr %47, i64 6400
  %v603 = getelementptr i8, ptr %114, i64 %v21
  %v604 = load i8, ptr %v603, align 1
  %v605 = zext i8 %v604 to i32
  %v610 = lshr i32 %v605, 3
  %v613 = and i32 %v610, 15
  %v614 = and i32 %v605, 7
  %v615 = icmp eq i32 %v613, 0
  br i1 %v615, label %bb350, label %bb354

bb350:                                            ; preds = %bb349
  %v1778 = uitofp nneg i32 %v614 to float
  %v1779 = fmul nnan float %v1778, f0x3B000000
  %v1781 = fneg float %v1779
  %v178054 = icmp slt i8 %v604, 0
  %v1782 = select i1 %v178054, float %v1781, float %v1779
  br label %bb355

bb354:                                            ; preds = %bb349
  %v604.signext = sext i8 %v604 to i32
  %v609 = and i32 %v604.signext, -2147483648
  %v1783 = shl nuw nsw i32 %v613, 23
  %v1786 = add nuw nsw i32 %v1783, 1006632960
  %v1790 = shl nuw nsw i32 %v614, 20
  %115 = or disjoint i32 %v1786, %v1790
  %v1791 = or disjoint i32 %115, %v609
  %v1792 = bitcast i32 %v1791 to float
  br label %bb355

bb355:                                            ; preds = %bb354, %bb350
  %v1793 = phi float [ %v1782, %bb350 ], [ %v1792, %bb354 ]
  %v1794 = fmul float %v9, %v1793
  %v616 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 100), align 4
  %v617 = fmul float %v1794, %v616
  %116 = getelementptr i8, ptr %47, i64 6656
  %v620 = getelementptr i8, ptr %116, i64 %v21
  %v621 = load i8, ptr %v620, align 1
  %v622 = zext i8 %v621 to i32
  %v627 = lshr i32 %v622, 3
  %v630 = and i32 %v627, 15
  %v631 = and i32 %v622, 7
  %v632 = icmp eq i32 %v630, 0
  br i1 %v632, label %bb356, label %bb360

bb356:                                            ; preds = %bb355
  %v1796 = uitofp nneg i32 %v631 to float
  %v1797 = fmul nnan float %v1796, f0x3B000000
  %v1799 = fneg float %v1797
  %v179855 = icmp slt i8 %v621, 0
  %v1800 = select i1 %v179855, float %v1799, float %v1797
  br label %bb361

bb360:                                            ; preds = %bb355
  %v621.signext = sext i8 %v621 to i32
  %v626 = and i32 %v621.signext, -2147483648
  %v1801 = shl nuw nsw i32 %v630, 23
  %v1804 = add nuw nsw i32 %v1801, 1006632960
  %v1808 = shl nuw nsw i32 %v631, 20
  %117 = or disjoint i32 %v1804, %v1808
  %v1809 = or disjoint i32 %117, %v626
  %v1810 = bitcast i32 %v1809 to float
  br label %bb361

bb361:                                            ; preds = %bb360, %bb356
  %v1811 = phi float [ %v1800, %bb356 ], [ %v1810, %bb360 ]
  %v1812 = fmul float %v9, %v1811
  %v633 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 104), align 4
  %v634 = fmul float %v1812, %v633
  %118 = getelementptr i8, ptr %47, i64 6912
  %v637 = getelementptr i8, ptr %118, i64 %v21
  %v638 = load i8, ptr %v637, align 1
  %v639 = zext i8 %v638 to i32
  %v644 = lshr i32 %v639, 3
  %v647 = and i32 %v644, 15
  %v648 = and i32 %v639, 7
  %v649 = icmp eq i32 %v647, 0
  br i1 %v649, label %bb362, label %bb366

bb362:                                            ; preds = %bb361
  %v1814 = uitofp nneg i32 %v648 to float
  %v1815 = fmul nnan float %v1814, f0x3B000000
  %v1817 = fneg float %v1815
  %v181656 = icmp slt i8 %v638, 0
  %v1818 = select i1 %v181656, float %v1817, float %v1815
  br label %bb367

bb366:                                            ; preds = %bb361
  %v638.signext = sext i8 %v638 to i32
  %v643 = and i32 %v638.signext, -2147483648
  %v1819 = shl nuw nsw i32 %v647, 23
  %v1822 = add nuw nsw i32 %v1819, 1006632960
  %v1826 = shl nuw nsw i32 %v648, 20
  %119 = or disjoint i32 %v1822, %v1826
  %v1827 = or disjoint i32 %119, %v643
  %v1828 = bitcast i32 %v1827 to float
  br label %bb367

bb367:                                            ; preds = %bb366, %bb362
  %v1829 = phi float [ %v1818, %bb362 ], [ %v1828, %bb366 ]
  %v1830 = fmul float %v9, %v1829
  %v650 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 108), align 4
  %v651 = fmul float %v1830, %v650
  %120 = getelementptr i8, ptr %47, i64 7168
  %v654 = getelementptr i8, ptr %120, i64 %v21
  %v655 = load i8, ptr %v654, align 1
  %v656 = zext i8 %v655 to i32
  %v661 = lshr i32 %v656, 3
  %v664 = and i32 %v661, 15
  %v665 = and i32 %v656, 7
  %v666 = icmp eq i32 %v664, 0
  br i1 %v666, label %bb368, label %bb372

bb368:                                            ; preds = %bb367
  %v1832 = uitofp nneg i32 %v665 to float
  %v1833 = fmul nnan float %v1832, f0x3B000000
  %v1835 = fneg float %v1833
  %v183457 = icmp slt i8 %v655, 0
  %v1836 = select i1 %v183457, float %v1835, float %v1833
  br label %bb373

bb372:                                            ; preds = %bb367
  %v655.signext = sext i8 %v655 to i32
  %v660 = and i32 %v655.signext, -2147483648
  %v1837 = shl nuw nsw i32 %v664, 23
  %v1840 = add nuw nsw i32 %v1837, 1006632960
  %v1844 = shl nuw nsw i32 %v665, 20
  %121 = or disjoint i32 %v1840, %v1844
  %v1845 = or disjoint i32 %121, %v660
  %v1846 = bitcast i32 %v1845 to float
  br label %bb373

bb373:                                            ; preds = %bb372, %bb368
  %v1847 = phi float [ %v1836, %bb368 ], [ %v1846, %bb372 ]
  %v1848 = fmul float %v9, %v1847
  %v667 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 112), align 4
  %v668 = fmul float %v1848, %v667
  %122 = getelementptr i8, ptr %47, i64 7424
  %v671 = getelementptr i8, ptr %122, i64 %v21
  %v672 = load i8, ptr %v671, align 1
  %v673 = zext i8 %v672 to i32
  %v678 = lshr i32 %v673, 3
  %v681 = and i32 %v678, 15
  %v682 = and i32 %v673, 7
  %v683 = icmp eq i32 %v681, 0
  br i1 %v683, label %bb374, label %bb378

bb374:                                            ; preds = %bb373
  %v1850 = uitofp nneg i32 %v682 to float
  %v1851 = fmul nnan float %v1850, f0x3B000000
  %v1853 = fneg float %v1851
  %v185258 = icmp slt i8 %v672, 0
  %v1854 = select i1 %v185258, float %v1853, float %v1851
  br label %bb379

bb378:                                            ; preds = %bb373
  %v672.signext = sext i8 %v672 to i32
  %v677 = and i32 %v672.signext, -2147483648
  %v1855 = shl nuw nsw i32 %v681, 23
  %v1858 = add nuw nsw i32 %v1855, 1006632960
  %v1862 = shl nuw nsw i32 %v682, 20
  %123 = or disjoint i32 %v1858, %v1862
  %v1863 = or disjoint i32 %123, %v677
  %v1864 = bitcast i32 %v1863 to float
  br label %bb379

bb379:                                            ; preds = %bb378, %bb374
  %v1865 = phi float [ %v1854, %bb374 ], [ %v1864, %bb378 ]
  %v1866 = fmul float %v9, %v1865
  %v684 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 116), align 4
  %v685 = fmul float %v1866, %v684
  %124 = getelementptr i8, ptr %47, i64 7680
  %v688 = getelementptr i8, ptr %124, i64 %v21
  %v689 = load i8, ptr %v688, align 1
  %v690 = zext i8 %v689 to i32
  %v695 = lshr i32 %v690, 3
  %v698 = and i32 %v695, 15
  %v699 = and i32 %v690, 7
  %v700 = icmp eq i32 %v698, 0
  br i1 %v700, label %bb380, label %bb384

bb380:                                            ; preds = %bb379
  %v1868 = uitofp nneg i32 %v699 to float
  %v1869 = fmul nnan float %v1868, f0x3B000000
  %v1871 = fneg float %v1869
  %v187059 = icmp slt i8 %v689, 0
  %v1872 = select i1 %v187059, float %v1871, float %v1869
  br label %bb385

bb384:                                            ; preds = %bb379
  %v689.signext = sext i8 %v689 to i32
  %v694 = and i32 %v689.signext, -2147483648
  %v1873 = shl nuw nsw i32 %v698, 23
  %v1876 = add nuw nsw i32 %v1873, 1006632960
  %v1880 = shl nuw nsw i32 %v699, 20
  %125 = or disjoint i32 %v1876, %v1880
  %v1881 = or disjoint i32 %125, %v694
  %v1882 = bitcast i32 %v1881 to float
  br label %bb385

bb385:                                            ; preds = %bb384, %bb380
  %v1883 = phi float [ %v1872, %bb380 ], [ %v1882, %bb384 ]
  %v1884 = fmul float %v9, %v1883
  %v701 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 120), align 4
  %v702 = fmul float %v1884, %v701
  %126 = getelementptr i8, ptr %47, i64 7936
  %v705 = getelementptr i8, ptr %126, i64 %v21
  %v706 = load i8, ptr %v705, align 1
  %v707 = zext i8 %v706 to i32
  %v712 = lshr i32 %v707, 3
  %v715 = and i32 %v712, 15
  %v716 = and i32 %v707, 7
  %v717 = icmp eq i32 %v715, 0
  br i1 %v717, label %bb386, label %bb390

bb386:                                            ; preds = %bb385
  %v1886 = uitofp nneg i32 %v716 to float
  %v1887 = fmul nnan float %v1886, f0x3B000000
  %v1889 = fneg float %v1887
  %v188860 = icmp slt i8 %v706, 0
  %v1890 = select i1 %v188860, float %v1889, float %v1887
  br label %bb391

bb390:                                            ; preds = %bb385
  %v706.signext = sext i8 %v706 to i32
  %v711 = and i32 %v706.signext, -2147483648
  %v1891 = shl nuw nsw i32 %v715, 23
  %v1894 = add nuw nsw i32 %v1891, 1006632960
  %v1898 = shl nuw nsw i32 %v716, 20
  %127 = or disjoint i32 %v1894, %v1898
  %v1899 = or disjoint i32 %127, %v711
  %v1900 = bitcast i32 %v1899 to float
  br label %bb391

bb391:                                            ; preds = %bb390, %bb386
  %v1901 = phi float [ %v1890, %bb386 ], [ %v1900, %bb390 ]
  %v1902 = fmul float %v9, %v1901
  %v718 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 124), align 4
  %v719 = fmul float %v1902, %v718
  %v720 = fadd float %v192, %v328
  %v721 = fadd float %v720, %v464
  %v722 = fadd float %v721, %v600
  %v723 = fadd float %v209, %v345
  %v724 = fadd float %v723, %v481
  %v725 = fadd float %v724, %v617
  %v726 = fadd float %v226, %v362
  %v727 = fadd float %v726, %v498
  %v728 = fadd float %v727, %v634
  %v729 = fadd float %v243, %v379
  %v730 = fadd float %v729, %v515
  %v731 = fadd float %v730, %v651
  %v732 = fadd float %v260, %v396
  %v733 = fadd float %v732, %v532
  %v734 = fadd float %v733, %v668
  %v735 = fadd float %v277, %v413
  %v736 = fadd float %v735, %v549
  %v737 = fadd float %v736, %v685
  %v738 = fadd float %v294, %v430
  %v739 = fadd float %v738, %v566
  %v740 = fadd float %v739, %v702
  %v741 = fadd float %v311, %v447
  %v742 = fadd float %v741, %v583
  %v743 = fadd float %v742, %v719
  %v744 = fadd float %v722, %v725
  %v745 = fadd float %v734, %v737
  %v746 = fadd float %v744, %v745
  %v747 = fadd float %v728, %v731
  %v748 = fadd float %v740, %v743
  %v749 = fadd float %v747, %v748
  %v750 = fadd float %v746, %v749
  %v752 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_23b72fdefd237786_0, i64 8), align 4
  %v753 = fmul float %v10673, %v752
  %v754 = fadd float %v753, %v750
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #8
  %v756 = add nsw i32 %v10774, 1
  %exitcond = icmp eq i32 %v10774, %v10811
  br i1 %exitcond, label %bb78, label %bb16
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare float @llvm.nvvm.shfl.sync.bfly.f32(i32, float, i32, i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare float @llvm.nvvm.shfl.sync.idx.f32(i32, float, i32, i32) #3

; Function Attrs: nofree nosync nounwind memory(none)
declare noundef i32 @__nvvm_reflect(ptr noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fma.rn.ftz.f(float, float, float) #5

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.saturate.ftz.f(float) #5

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.saturate.f(float) #5

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fma.rm.ftz.f(float, float, float) #5

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fma.rm.f(float, float, float) #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.ex2.approx.ftz.f32(float) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #7

attributes #0 = { convergent nounwind }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }
attributes #3 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #4 = { nofree nosync nounwind memory(none) "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #7 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent }
attributes #9 = { nounwind }

!llvm.ident = !{!0}
!nvvmir.version = !{!1}

!0 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!1 = !{i32 2, i32 0}
