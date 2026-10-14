; ModuleID = '/home/r/benchmarks/inference.rs/cuda-oxide-gdn-2026-10-15/source/crates/backend/cuda/kernels/oxide-gdn/oxide_gdn_simt.ll'
source_filename = "oxide_gdn_simt"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@__shared_mem_a858d0f5fb2dcc30_2 = local_unnamed_addr addrspace(3) global [128 x float] zeroinitializer, align 4
@__shared_mem_a858d0f5fb2dcc30_1 = local_unnamed_addr addrspace(3) global [8 x float] zeroinitializer, align 4
@__shared_mem_a858d0f5fb2dcc30_0 = local_unnamed_addr addrspace(3) global [256 x float] zeroinitializer, align 4
@llvm.used = appending global [1 x ptr] [ptr @gdn_f32_128], section "llvm.metadata"

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #1

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare float @llvm.nvvm.shfl.sync.bfly.f32(i32, float, i32, i32) #2

; Function Attrs: convergent norecurse nounwind
define ptx_kernel void @gdn_f32_128(ptr nofree captures(none) %v0, ptr nofree writeonly captures(none) %v1, ptr nofree readonly captures(none) %v2, ptr nofree readonly captures(none) %v3, ptr nofree readonly captures(none) %v4, ptr nofree readonly captures(none) %v5, ptr nofree readonly captures(none) %v6, i32 %v7) #3 {
entry:
  %v16 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5
  %v17 = zext nneg i32 %v16 to i64
  %v181 = lshr i64 %v17, 3
  %v19 = and i64 %v17, 7
  %v20 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5
  %v21 = zext nneg i32 %v20 to i64
  %v222 = lshr i64 %v21, 4
  %v23 = shl nuw nsw i64 %v21, 3
  %v24 = and i64 %v23, 120
  %v25 = or disjoint i64 %v24, %v19
  %v2614 = udiv i32 %v20, 48
  %v26.zext = zext nneg i32 %v2614 to i64
  %v27 = zext i32 %v7 to i64
  %v28 = getelementptr inbounds nuw i8, ptr %v6, i64 4
  %v29 = load i32, ptr %v28, align 4
  %v30 = load i32, ptr %v6, align 4
  %v31 = getelementptr inbounds nuw i8, ptr %v6, i64 8
  %v32 = load i32, ptr %v31, align 4
  %v33 = add i32 %v32, %v30
  %v36 = sub i32 0, %v33
  %v34.inv = icmp slt i32 %v33, 0
  %narrow = select i1 %v34.inv, i32 %v36, i32 0
  %v38 = sext i32 %narrow to i64
  %v39 = icmp slt i32 %v29, 1
  %v41 = sext i32 %v29 to i64
  %v41.v27 = tail call i64 @llvm.umin.i64(i64 %v41, i64 %v27)
  %invariant.gep = getelementptr [4 x i8], ptr %v1, i64 %v25
  %v47.not17 = icmp samesign ult i64 %v181, %v27
  br i1 %v47.not17, label %bb13.lr.ph, label %bb14

bb13.lr.ph:                                       ; preds = %entry
  %v49 = mul nuw nsw i64 %v222, %v27
  br label %bb13

bb13:                                             ; preds = %bb13.lr.ph, %bb13
  %v4618 = phi i64 [ %v181, %bb13.lr.ph ], [ %v54, %bb13 ]
  %v50 = add nuw nsw i64 %v4618, %v49
  %.idx = shl i64 %v50, 9
  %gep = getelementptr i8, ptr %invariant.gep, i64 %.idx
  store float 0.000000e+00, ptr %gep, align 4
  %v54 = add nuw nsw i64 %v4618, 16
  %v47.not = icmp samesign ult i64 %v54, %v27
  br i1 %v47.not, label %bb13, label %bb14

bb14:                                             ; preds = %bb13, %entry
  %0 = shl nuw nsw i64 %v222, 16
  %1 = shl nuw nsw i64 %v181, 9
  %2 = getelementptr inbounds nuw i8, ptr %v0, i64 %0
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 %1
  %v59 = getelementptr inbounds nuw [4 x i8], ptr %3, i64 %v25
  %v60 = load float, ptr %v59, align 4
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 8192
  %v64 = getelementptr inbounds nuw [4 x i8], ptr %4, i64 %v25
  %v65 = load float, ptr %v64, align 4
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16384
  %v69 = getelementptr inbounds nuw [4 x i8], ptr %5, i64 %v25
  %v70 = load float, ptr %v69, align 4
  %6 = getelementptr inbounds nuw i8, ptr %3, i64 24576
  %v74 = getelementptr inbounds nuw [4 x i8], ptr %6, i64 %v25
  %v75 = load float, ptr %v74, align 4
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 32768
  %v79 = getelementptr inbounds nuw [4 x i8], ptr %7, i64 %v25
  %v80 = load float, ptr %v79, align 4
  %8 = getelementptr inbounds nuw i8, ptr %3, i64 40960
  %v84 = getelementptr inbounds nuw [4 x i8], ptr %8, i64 %v25
  %v85 = load float, ptr %v84, align 4
  %9 = getelementptr inbounds nuw i8, ptr %3, i64 49152
  %v89 = getelementptr inbounds nuw [4 x i8], ptr %9, i64 %v25
  %v90 = load float, ptr %v89, align 4
  %10 = getelementptr inbounds nuw i8, ptr %3, i64 57344
  %v94 = getelementptr inbounds nuw [4 x i8], ptr %10, i64 %v25
  %v95 = load float, ptr %v94, align 4
  %invariant.gep19 = getelementptr [4 x i8], ptr %v2, i64 %v25
  %invariant.gep21 = getelementptr [4 x i8], ptr %v3, i64 %v181
  %v1051127 = icmp ule i64 %v41.v27, %v38
  %v105.not28 = select i1 %v39, i1 true, i1 %v1051127
  br i1 %v105.not28, label %bb45, label %bb16.lr.ph

bb16.lr.ph:                                       ; preds = %bb14
  %v116 = add nuw nsw i64 %v222, 32
  %v129 = add nuw nsw i64 %v26.zext, 16
  %v266 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_0, i64 %v17
  %v267 = and i64 %v17, 1016
  %11 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_0, i64 %v267
  %12 = getelementptr inbounds nuw i8, ptr addrspace(3) %11, i64 512
  %v270 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %12, i64 %v19
  %v272 = icmp samesign ugt i32 %v16, 31
  %v279 = getelementptr inbounds nuw i8, ptr addrspace(3) %v266, i64 128
  %v284 = getelementptr inbounds nuw i8, ptr addrspace(3) %v266, i64 256
  %v289 = getelementptr inbounds nuw i8, ptr addrspace(3) %v266, i64 384
  %v299 = getelementptr inbounds nuw i8, ptr addrspace(3) %v266, i64 640
  %v304 = getelementptr inbounds nuw i8, ptr addrspace(3) %v266, i64 768
  %v309 = getelementptr inbounds nuw i8, ptr addrspace(3) %v266, i64 896
  %v390 = icmp samesign ugt i32 %v16, 7
  %v319 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_1, i64 %v19
  %v357 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_2, i64 %v17
  %v365 = getelementptr inbounds nuw i8, ptr addrspace(3) %v357, i64 128
  %v370 = getelementptr inbounds nuw i8, ptr addrspace(3) %v357, i64 256
  %v375 = getelementptr inbounds nuw i8, ptr addrspace(3) %v357, i64 384
  %v381 = mul nuw nsw i64 %v222, %v27
  br label %bb16

bb16:                                             ; preds = %bb16.lr.ph, %bb44
  %v10437 = phi float [ %v95, %bb16.lr.ph ], [ %v347, %bb44 ]
  %v10336 = phi float [ %v90, %bb16.lr.ph ], [ %v344, %bb44 ]
  %v10235 = phi float [ %v85, %bb16.lr.ph ], [ %v341, %bb44 ]
  %v10134 = phi float [ %v80, %bb16.lr.ph ], [ %v338, %bb44 ]
  %v10033 = phi float [ %v75, %bb16.lr.ph ], [ %v335, %bb44 ]
  %v9932 = phi float [ %v70, %bb16.lr.ph ], [ %v332, %bb44 ]
  %v9831 = phi float [ %v65, %bb16.lr.ph ], [ %v329, %bb44 ]
  %v9730 = phi float [ %v60, %bb16.lr.ph ], [ %v326, %bb44 ]
  %v9629 = phi i64 [ %v38, %bb16.lr.ph ], [ %v386, %bb44 ]
  %v107 = mul i64 %v9629, 48
  %v108 = add i64 %v107, %v222
  %v109 = getelementptr inbounds [4 x i8], ptr %v5, i64 %v108
  %v110 = load float, ptr %v109, align 4
  %v113 = getelementptr inbounds [4 x i8], ptr %v4, i64 %v108
  %v114 = load float, ptr %v113, align 4
  %v115 = mul i64 %v9629, 80
  %v117 = add i64 %v116, %v115
  %.idx12 = shl i64 %v117, 9
  %gep20 = getelementptr i8, ptr %invariant.gep19, i64 %.idx12
  %v121 = load float, ptr %gep20, align 4
  %v122 = shl i64 %v9629, 5
  %v123 = add i64 %v122, %v26.zext
  %.idx15 = shl i64 %v123, 9
  %gep22 = getelementptr i8, ptr %invariant.gep21, i64 %.idx15
  %v127 = load float, ptr %gep22, align 4
  %v130 = add i64 %v129, %v122
  %.idx16 = shl i64 %v130, 9
  %gep24 = getelementptr i8, ptr %invariant.gep21, i64 %.idx16
  %v134 = load float, ptr %gep24, align 4
  %v135 = fmul float %v9730, %v110
  %v136 = fmul float %v135, %v134
  %v142 = getelementptr i8, ptr %gep22, i64 64
  %v143 = load float, ptr %v142, align 4
  %v150 = getelementptr i8, ptr %gep24, i64 64
  %v151 = load float, ptr %v150, align 4
  %v152 = fmul float %v9831, %v110
  %v153 = fmul float %v152, %v151
  %v159 = getelementptr i8, ptr %gep22, i64 128
  %v160 = load float, ptr %v159, align 4
  %v167 = getelementptr i8, ptr %gep24, i64 128
  %v168 = load float, ptr %v167, align 4
  %v169 = fmul float %v9932, %v110
  %v170 = fmul float %v169, %v168
  %v176 = getelementptr i8, ptr %gep22, i64 192
  %v177 = load float, ptr %v176, align 4
  %v184 = getelementptr i8, ptr %gep24, i64 192
  %v185 = load float, ptr %v184, align 4
  %v186 = fmul float %v10033, %v110
  %v187 = fmul float %v186, %v185
  %v193 = getelementptr i8, ptr %gep22, i64 256
  %v194 = load float, ptr %v193, align 4
  %v201 = getelementptr i8, ptr %gep24, i64 256
  %v202 = load float, ptr %v201, align 4
  %v203 = fmul float %v10134, %v110
  %v204 = fmul float %v203, %v202
  %v210 = getelementptr i8, ptr %gep22, i64 320
  %v211 = load float, ptr %v210, align 4
  %v218 = getelementptr i8, ptr %gep24, i64 320
  %v219 = load float, ptr %v218, align 4
  %v220 = fmul float %v10235, %v110
  %v221 = fmul float %v220, %v219
  %v227 = getelementptr i8, ptr %gep22, i64 384
  %v228 = load float, ptr %v227, align 4
  %v235 = getelementptr i8, ptr %gep24, i64 384
  %v236 = load float, ptr %v235, align 4
  %v237 = fmul float %v10336, %v110
  %v238 = fmul float %v237, %v236
  %v244 = getelementptr i8, ptr %gep22, i64 448
  %v245 = load float, ptr %v244, align 4
  %v252 = getelementptr i8, ptr %gep24, i64 448
  %v253 = load float, ptr %v252, align 4
  %v254 = fmul float %v10437, %v110
  %v255 = fmul float %v254, %v253
  %v256 = fadd float %v136, 0.000000e+00
  %v257 = fadd float %v256, %v170
  %v258 = fadd float %v257, %v204
  %v259 = fadd float %v258, %v238
  %v260 = fadd float %v153, 0.000000e+00
  %v261 = fadd float %v260, %v187
  %v262 = fadd float %v261, %v221
  %v263 = fadd float %v262, %v255
  store float %v259, ptr addrspace(3) %v266, align 4
  store float %v263, ptr addrspace(3) %v270, align 4
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #5
  br i1 %v272, label %bb32, label %bb20

bb20:                                             ; preds = %bb16
  %v275 = load float, ptr addrspace(3) %v266, align 4
  %v280 = load float, ptr addrspace(3) %v279, align 4
  %v285 = load float, ptr addrspace(3) %v284, align 4
  %v290 = load float, ptr addrspace(3) %v289, align 4
  %v295 = load float, ptr addrspace(3) %v270, align 4
  %v300 = load float, ptr addrspace(3) %v299, align 4
  %v305 = load float, ptr addrspace(3) %v304, align 4
  %v310 = load float, ptr addrspace(3) %v309, align 4
  %v311 = fadd float %v275, %v280
  %v312 = fadd float %v285, %v311
  %v313 = fadd float %v290, %v312
  %v314 = fadd float %v295, %v313
  %v315 = fadd float %v300, %v314
  %v316 = fadd float %v305, %v315
  %v317 = fadd float %v310, %v316
  %v318 = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v317, i32 16, i32 31) #5
  %v387 = fadd float %v317, %v318
  %v388 = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v387, i32 8, i32 31) #5
  br i1 %v390, label %bb32, label %bb29

bb29:                                             ; preds = %bb20
  %v389 = fadd float %v387, %v388
  store float %v389, ptr addrspace(3) %v319, align 4
  br label %bb32

bb32:                                             ; preds = %bb29, %bb20, %bb16
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #5
  %v322 = load float, ptr addrspace(3) %v319, align 4
  %v323 = fsub float %v121, %v322
  %v324 = fmul float %v114, %v323
  %v325 = fmul float %v134, %v324
  %v326 = fadd float %v135, %v325
  %v327 = fmul float %v127, %v326
  %v328 = fmul float %v151, %v324
  %v329 = fadd float %v152, %v328
  %v330 = fmul float %v143, %v329
  %v331 = fmul float %v168, %v324
  %v332 = fadd float %v169, %v331
  %v333 = fmul float %v160, %v332
  %v334 = fmul float %v185, %v324
  %v335 = fadd float %v186, %v334
  %v336 = fmul float %v177, %v335
  %v337 = fmul float %v202, %v324
  %v338 = fadd float %v203, %v337
  %v339 = fmul float %v194, %v338
  %v340 = fmul float %v219, %v324
  %v341 = fadd float %v220, %v340
  %v342 = fmul float %v211, %v341
  %v343 = fmul float %v236, %v324
  %v344 = fadd float %v237, %v343
  %v345 = fmul float %v228, %v344
  %v346 = fmul float %v253, %v324
  %v347 = fadd float %v254, %v346
  %v348 = fmul float %v245, %v347
  %v349 = fadd float %v327, 0.000000e+00
  %v350 = fadd float %v330, %v349
  %v351 = fadd float %v333, %v350
  %v352 = fadd float %v336, %v351
  %v353 = fadd float %v339, %v352
  %v354 = fadd float %v342, %v353
  %v355 = fadd float %v345, %v354
  %v356 = fadd float %v348, %v355
  store float %v356, ptr addrspace(3) %v357, align 4
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #5
  br i1 %v272, label %bb44, label %bb37

bb37:                                             ; preds = %bb32
  %v361 = load float, ptr addrspace(3) %v357, align 4
  %v366 = load float, ptr addrspace(3) %v365, align 4
  %v371 = load float, ptr addrspace(3) %v370, align 4
  %v376 = load float, ptr addrspace(3) %v375, align 4
  %v377 = fadd float %v361, %v366
  %v378 = fadd float %v371, %v377
  %v379 = fadd float %v376, %v378
  %v380 = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v379, i32 16, i32 31) #5
  %v392 = fadd float %v379, %v380
  %v393 = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v392, i32 8, i32 31) #5
  br i1 %v390, label %bb44, label %bb42

bb42:                                             ; preds = %bb37
  %v394 = fadd float %v392, %v393
  %v382 = add i64 %v9629, %v381
  %.idx13 = shl i64 %v382, 9
  %gep26 = getelementptr i8, ptr %invariant.gep, i64 %.idx13
  store float %v394, ptr %gep26, align 4
  br label %bb44

bb44:                                             ; preds = %bb42, %bb37, %bb32
  %v386 = add nuw nsw i64 %v9629, 1
  %v10511.not = icmp ugt i64 %v41.v27, %v386
  br i1 %v10511.not, label %bb16, label %bb45

bb45:                                             ; preds = %bb44, %bb14
  %v97.lcssa = phi float [ %v60, %bb14 ], [ %v326, %bb44 ]
  %v98.lcssa = phi float [ %v65, %bb14 ], [ %v329, %bb44 ]
  %v99.lcssa = phi float [ %v70, %bb14 ], [ %v332, %bb44 ]
  %v100.lcssa = phi float [ %v75, %bb14 ], [ %v335, %bb44 ]
  %v101.lcssa = phi float [ %v80, %bb14 ], [ %v338, %bb44 ]
  %v102.lcssa = phi float [ %v85, %bb14 ], [ %v341, %bb44 ]
  %v103.lcssa = phi float [ %v90, %bb14 ], [ %v344, %bb44 ]
  %v104.lcssa = phi float [ %v95, %bb14 ], [ %v347, %bb44 ]
  store float %v97.lcssa, ptr %v59, align 4
  store float %v98.lcssa, ptr %v64, align 4
  store float %v99.lcssa, ptr %v69, align 4
  store float %v100.lcssa, ptr %v74, align 4
  store float %v101.lcssa, ptr %v79, align 4
  store float %v102.lcssa, ptr %v84, align 4
  store float %v103.lcssa, ptr %v89, align 4
  store float %v104.lcssa, ptr %v94, align 4
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #4

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { convergent nocallback nounwind }
attributes #2 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #3 = { convergent norecurse nounwind }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent }
