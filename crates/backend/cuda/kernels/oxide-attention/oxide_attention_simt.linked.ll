; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@llvm.used = appending global [1 x ptr] [ptr @attention_fp8_256], section "llvm.metadata"
@__shared_mem_23b72fdefd237786_2 = addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_23b72fdefd237786_1 = addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_23b72fdefd237786_0 = addrspace(3) global [4 x float] zeroinitializer, align 4
@.str = private unnamed_addr constant [11 x i8] c"__CUDA_FTZ\00", align 1

; Function Attrs: convergent
define ptx_kernel void @attention_fp8_256(ptr %v0, ptr %v1, ptr %v2, ptr %v3, ptr %v4, ptr %v5, i32 %v6, i32 %v7, float %v8, float %v9) #0 {
entry:
  br label %bb0

bb0:                                              ; preds = %entry
  %v10 = phi ptr [ %v0, %entry ]
  %v11 = phi ptr [ %v1, %entry ]
  %v12 = phi ptr [ %v2, %entry ]
  %v13 = phi ptr [ %v3, %entry ]
  %v14 = phi ptr [ %v4, %entry ]
  %v15 = phi ptr [ %v5, %entry ]
  %v16 = phi i32 [ %v6, %entry ]
  %v17 = phi i32 [ %v7, %entry ]
  %v18 = phi float [ %v8, %entry ]
  %v19 = phi float [ %v9, %entry ]
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1

bb1:                                              ; preds = %bb0
  %v21 = zext i32 %v20 to i64
  %v22 = udiv i64 %v21, 32
  %v23 = urem i64 %v21, 32
  %v24 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2

bb2:                                              ; preds = %bb1
  %v25 = zext i32 %v24 to i64
  %v26 = udiv i64 %v25, 24
  %v27 = urem i64 %v25, 24
  %v28 = load i32, ptr %v13, align 4
  %v29 = trunc i64 %v26 to i32
  %v30 = add i32 %v28, %v29
  %v31 = getelementptr inbounds i32, ptr %v13, i64 2
  %v32 = load i32, ptr %v31, align 4
  %v33 = add i32 %v30, %v32
  %v34 = getelementptr inbounds i32, ptr %v13, i64 1
  %v35 = load i32, ptr %v34, align 4
  %v36 = icmp sge i32 %v29, %v35
  %v37 = xor i1 %v36, true
  br i1 %v37, label %bb3, label %bb4

bb3:                                              ; preds = %bb2
  %v38 = icmp slt i32 %v33, 0
  %v39 = xor i1 %v38, true
  br i1 %v39, label %bb5, label %bb4

bb4:                                              ; preds = %bb3, %bb2
  %v40 = mul i64 %v25, 256
  %v41 = add i64 %v40, %v21
  %v42 = getelementptr inbounds float, ptr %v15, i64 %v41
  store float 0.000000e+00, ptr %v42, align 4
  br label %bb80

bb5:                                              ; preds = %bb3
  %v43 = icmp sgt i32 %v17, 0
  %v44 = xor i1 %v43, true
  br i1 %v44, label %bb8, label %bb6

bb6:                                              ; preds = %bb5
  %v45 = add i32 %v33, 1
  %v46 = icmp sgt i32 %v45, %v17
  %v47 = xor i1 %v46, true
  br i1 %v47, label %bb8, label %bb7

bb7:                                              ; preds = %bb6
  %v48 = sub i32 %v45, %v17
  br label %bb9

bb8:                                              ; preds = %bb6, %bb5
  br label %bb9

bb9:                                              ; preds = %bb8, %bb7
  %v49 = phi i32 [ %v48, %bb7 ], [ 0, %bb8 ]
  %v50 = and i64 %v23, 15
  %v51 = mul i64 %v25, 256
  %v52 = mul i64 %v50, 16
  %v53 = add i64 %v51, %v52
  %v54 = getelementptr inbounds float, ptr %v10, i64 %v53
  %v55 = load float, ptr %v54, align 4
  %v56 = add i64 %v53, 1
  %v57 = getelementptr inbounds float, ptr %v10, i64 %v56
  %v58 = load float, ptr %v57, align 4
  %v59 = add i64 %v53, 2
  %v60 = getelementptr inbounds float, ptr %v10, i64 %v59
  %v61 = load float, ptr %v60, align 4
  %v62 = add i64 %v53, 3
  %v63 = getelementptr inbounds float, ptr %v10, i64 %v62
  %v64 = load float, ptr %v63, align 4
  %v65 = add i64 %v53, 4
  %v66 = getelementptr inbounds float, ptr %v10, i64 %v65
  %v67 = load float, ptr %v66, align 4
  %v68 = add i64 %v53, 5
  %v69 = getelementptr inbounds float, ptr %v10, i64 %v68
  %v70 = load float, ptr %v69, align 4
  %v71 = add i64 %v53, 6
  %v72 = getelementptr inbounds float, ptr %v10, i64 %v71
  %v73 = load float, ptr %v72, align 4
  %v74 = add i64 %v53, 7
  %v75 = getelementptr inbounds float, ptr %v10, i64 %v74
  %v76 = load float, ptr %v75, align 4
  %v77 = add i64 %v53, 8
  %v78 = getelementptr inbounds float, ptr %v10, i64 %v77
  %v79 = load float, ptr %v78, align 4
  %v80 = add i64 %v53, 9
  %v81 = getelementptr inbounds float, ptr %v10, i64 %v80
  %v82 = load float, ptr %v81, align 4
  %v83 = add i64 %v53, 10
  %v84 = getelementptr inbounds float, ptr %v10, i64 %v83
  %v85 = load float, ptr %v84, align 4
  %v86 = add i64 %v53, 11
  %v87 = getelementptr inbounds float, ptr %v10, i64 %v86
  %v88 = load float, ptr %v87, align 4
  %v89 = add i64 %v53, 12
  %v90 = getelementptr inbounds float, ptr %v10, i64 %v89
  %v91 = load float, ptr %v90, align 4
  %v92 = add i64 %v53, 13
  %v93 = getelementptr inbounds float, ptr %v10, i64 %v92
  %v94 = load float, ptr %v93, align 4
  %v95 = add i64 %v53, 14
  %v96 = getelementptr inbounds float, ptr %v10, i64 %v95
  %v97 = load float, ptr %v96, align 4
  %v98 = add i64 %v53, 15
  %v99 = getelementptr inbounds float, ptr %v10, i64 %v98
  %v100 = load float, ptr %v99, align 4
  %v101 = icmp eq i64 %v21, 0
  br i1 %v101, label %bb10, label %bb13

bb10:                                             ; preds = %bb9
  %v102 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_0, i64 0
  br label %bb11

bb11:                                             ; preds = %bb10
  store float -1.000000e+30, ptr addrspace(3) %v102, align 4
  %v103 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_0, i64 1
  br label %bb12

bb12:                                             ; preds = %bb11
  store float 0.000000e+00, ptr addrspace(3) %v103, align 4
  br label %bb13

bb13:                                             ; preds = %bb12, %bb9
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb14

bb14:                                             ; preds = %bb13
  %v105 = sdiv i32 %v49, 32
  br label %bb15

bb15:                                             ; preds = %bb77, %bb14
  %v106 = phi float [ 0.000000e+00, %bb14 ], [ %v754, %bb77 ]
  %v107 = phi i32 [ %v105, %bb14 ], [ %v756, %bb77 ]
  %v108 = sdiv i32 %v33, 32
  %v109 = add i32 %v108, 1
  %v110 = icmp slt i32 %v107, %v109
  %v111 = xor i1 %v110, true
  br i1 %v111, label %bb78, label %bb16

bb16:                                             ; preds = %bb15
  %v112 = sext i32 %v107 to i64
  %v113 = getelementptr inbounds i32, ptr %v14, i64 %v112
  %v114 = load i32, ptr %v113, align 4
  %v115 = sext i32 %v114 to i64
  %v116 = udiv i64 %v27, 6
  %v117 = zext i32 %v16 to i64
  %v118 = mul i64 %v116, %v117
  %v119 = mul i64 %v115, 32
  %v120 = add i64 %v118, %v119
  %v121 = mul i64 %v120, 256
  %v122 = mul i64 %v22, 2
  %v123 = udiv i64 %v23, 16
  %v124 = add i64 %v122, %v123
  br label %bb17

bb17:                                             ; preds = %bb26, %bb16
  %v125 = phi i64 [ %v124, %bb16 ], [ %v154, %bb26 ]
  %v126 = icmp ult i64 %v125, 32
  %v127 = xor i1 %v126, true
  br i1 %v127, label %bb27, label %bb18

bb18:                                             ; preds = %bb17
  %v128 = mul i64 %v125, 256
  %v129 = add i64 %v121, %v128
  %v130 = add i64 %v129, %v52
  %v131 = getelementptr inbounds i8, ptr %v11, i64 %v130
  %v132 = load i8, ptr %v131, align 1
  %v133 = zext i8 %v132 to i32
  %v134 = and i32 %v133, 128
  %v135 = trunc i64 24 to i32
  %v136 = and i32 %v135, 31
  %v137 = shl i32 %v134, %v136
  %v138 = and i32 %v133, 127
  %v139 = trunc i64 3 to i32
  %v140 = and i32 %v139, 31
  %v141 = lshr i32 %v138, %v140
  %v142 = and i32 %v133, 7
  %v143 = icmp eq i32 %v141, 0
  br i1 %v143, label %bb81, label %bb85

bb19:                                             ; preds = %bb180
  %v144 = mul i32 %v107, 32
  %v145 = trunc i64 %v125 to i32
  %v146 = add i32 %v144, %v145
  %v147 = icmp sge i32 %v146, %v49
  %v148 = xor i1 %v147, true
  br i1 %v148, label %bb22, label %bb20

bb20:                                             ; preds = %bb19
  %v149 = icmp sle i32 %v146, %v33
  %v150 = xor i1 %v149, true
  br i1 %v150, label %bb23, label %bb21

bb21:                                             ; preds = %bb20
  %v151 = fmul float %v1282, 6.250000e-02
  br label %bb24

bb22:                                             ; preds = %bb19
  br label %bb23

bb23:                                             ; preds = %bb22, %bb20
  br label %bb24

bb24:                                             ; preds = %bb23, %bb21
  %v152 = phi float [ %v151, %bb21 ], [ -1.000000e+30, %bb23 ]
  %v153 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_1, i64 %v125
  br label %bb25

bb25:                                             ; preds = %bb24
  store float %v152, ptr addrspace(3) %v153, align 4
  br label %bb26

bb26:                                             ; preds = %bb180, %bb25
  %v154 = add i64 %v125, 16
  br label %bb17

bb27:                                             ; preds = %bb17
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb28

bb28:                                             ; preds = %bb27
  %v156 = icmp eq i64 %v22, 0
  br i1 %v156, label %bb29, label %bb42

bb29:                                             ; preds = %bb28
  %v157 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_1, i64 %v23
  br label %bb30

bb30:                                             ; preds = %bb29
  %v158 = load float, ptr addrspace(3) %v157, align 4
  %v159 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v158, i32 1, i32 31) #0
  br label %bb181

bb31:                                             ; preds = %bb190
  %v160 = load float, ptr addrspace(3) %v1308, align 4
  %v161 = fcmp ogt float %v1307, %v160
  %v162 = xor i1 %v161, true
  br i1 %v162, label %bb33, label %bb32

bb32:                                             ; preds = %bb31
  br label %bb34

bb33:                                             ; preds = %bb31
  br label %bb34

bb34:                                             ; preds = %bb33, %bb32
  %v163 = phi float [ %v1307, %bb32 ], [ %v160, %bb33 ]
  %v164 = fsub float %v160, %v163
  %v165 = call float @__nv_expf(float %v164) #0
  br label %bb191

bb35:                                             ; preds = %bb192
  store float %v1310, ptr addrspace(3) %v1311, align 4
  %v166 = trunc i64 %v23 to i32
  %v167 = and i32 %v166, 7
  %v168 = call float @llvm.nvvm.shfl.sync.idx.f32(i32 -1, float %v1310, i32 %v167, i32 31) #0
  br label %bb193

bb36:                                             ; preds = %bb199
  %v169 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_0, i64 0
  br label %bb37

bb37:                                             ; preds = %bb36
  store float %v163, ptr addrspace(3) %v169, align 4
  %v170 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_0, i64 1
  br label %bb38

bb38:                                             ; preds = %bb37
  %v171 = load float, ptr addrspace(3) %v170, align 4
  %v172 = fmul float %v171, %v165
  %v173 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_0, i64 1
  br label %bb39

bb39:                                             ; preds = %bb38
  %v174 = fadd float %v172, %v1326
  store float %v174, ptr addrspace(3) %v173, align 4
  %v175 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_0, i64 2
  br label %bb40

bb40:                                             ; preds = %bb39
  store float %v165, ptr addrspace(3) %v175, align 4
  br label %bb41

bb41:                                             ; preds = %bb199, %bb40
  br label %bb42

bb42:                                             ; preds = %bb41, %bb28
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb43

bb43:                                             ; preds = %bb42
  %v177 = add i64 %v121, %v21
  %v178 = getelementptr inbounds i8, ptr %v12, i64 %v177
  %v179 = load i8, ptr %v178, align 1
  %v180 = zext i8 %v179 to i32
  %v181 = and i32 %v180, 128
  %v182 = trunc i64 24 to i32
  %v183 = and i32 %v182, 31
  %v184 = shl i32 %v181, %v183
  %v185 = and i32 %v180, 127
  %v186 = trunc i64 3 to i32
  %v187 = and i32 %v186, 31
  %v188 = lshr i32 %v185, %v187
  %v189 = and i32 %v180, 7
  %v190 = icmp eq i32 %v188, 0
  br i1 %v190, label %bb200, label %bb204

bb44:                                             ; preds = %bb205
  %v191 = load float, ptr addrspace(3) %v1345, align 4
  %v192 = fmul float %v1344, %v191
  %v193 = add i64 %v121, 256
  %v194 = add i64 %v193, %v21
  %v195 = getelementptr inbounds i8, ptr %v12, i64 %v194
  %v196 = load i8, ptr %v195, align 1
  %v197 = zext i8 %v196 to i32
  %v198 = and i32 %v197, 128
  %v199 = trunc i64 24 to i32
  %v200 = and i32 %v199, 31
  %v201 = shl i32 %v198, %v200
  %v202 = and i32 %v197, 127
  %v203 = trunc i64 3 to i32
  %v204 = and i32 %v203, 31
  %v205 = lshr i32 %v202, %v204
  %v206 = and i32 %v197, 7
  %v207 = icmp eq i32 %v205, 0
  br i1 %v207, label %bb206, label %bb210

bb45:                                             ; preds = %bb211
  %v208 = load float, ptr addrspace(3) %v1363, align 4
  %v209 = fmul float %v1362, %v208
  %v210 = add i64 %v121, 512
  %v211 = add i64 %v210, %v21
  %v212 = getelementptr inbounds i8, ptr %v12, i64 %v211
  %v213 = load i8, ptr %v212, align 1
  %v214 = zext i8 %v213 to i32
  %v215 = and i32 %v214, 128
  %v216 = trunc i64 24 to i32
  %v217 = and i32 %v216, 31
  %v218 = shl i32 %v215, %v217
  %v219 = and i32 %v214, 127
  %v220 = trunc i64 3 to i32
  %v221 = and i32 %v220, 31
  %v222 = lshr i32 %v219, %v221
  %v223 = and i32 %v214, 7
  %v224 = icmp eq i32 %v222, 0
  br i1 %v224, label %bb212, label %bb216

bb46:                                             ; preds = %bb217
  %v225 = load float, ptr addrspace(3) %v1381, align 4
  %v226 = fmul float %v1380, %v225
  %v227 = add i64 %v121, 768
  %v228 = add i64 %v227, %v21
  %v229 = getelementptr inbounds i8, ptr %v12, i64 %v228
  %v230 = load i8, ptr %v229, align 1
  %v231 = zext i8 %v230 to i32
  %v232 = and i32 %v231, 128
  %v233 = trunc i64 24 to i32
  %v234 = and i32 %v233, 31
  %v235 = shl i32 %v232, %v234
  %v236 = and i32 %v231, 127
  %v237 = trunc i64 3 to i32
  %v238 = and i32 %v237, 31
  %v239 = lshr i32 %v236, %v238
  %v240 = and i32 %v231, 7
  %v241 = icmp eq i32 %v239, 0
  br i1 %v241, label %bb218, label %bb222

bb47:                                             ; preds = %bb223
  %v242 = load float, ptr addrspace(3) %v1399, align 4
  %v243 = fmul float %v1398, %v242
  %v244 = add i64 %v121, 1024
  %v245 = add i64 %v244, %v21
  %v246 = getelementptr inbounds i8, ptr %v12, i64 %v245
  %v247 = load i8, ptr %v246, align 1
  %v248 = zext i8 %v247 to i32
  %v249 = and i32 %v248, 128
  %v250 = trunc i64 24 to i32
  %v251 = and i32 %v250, 31
  %v252 = shl i32 %v249, %v251
  %v253 = and i32 %v248, 127
  %v254 = trunc i64 3 to i32
  %v255 = and i32 %v254, 31
  %v256 = lshr i32 %v253, %v255
  %v257 = and i32 %v248, 7
  %v258 = icmp eq i32 %v256, 0
  br i1 %v258, label %bb224, label %bb228

bb48:                                             ; preds = %bb229
  %v259 = load float, ptr addrspace(3) %v1417, align 4
  %v260 = fmul float %v1416, %v259
  %v261 = add i64 %v121, 1280
  %v262 = add i64 %v261, %v21
  %v263 = getelementptr inbounds i8, ptr %v12, i64 %v262
  %v264 = load i8, ptr %v263, align 1
  %v265 = zext i8 %v264 to i32
  %v266 = and i32 %v265, 128
  %v267 = trunc i64 24 to i32
  %v268 = and i32 %v267, 31
  %v269 = shl i32 %v266, %v268
  %v270 = and i32 %v265, 127
  %v271 = trunc i64 3 to i32
  %v272 = and i32 %v271, 31
  %v273 = lshr i32 %v270, %v272
  %v274 = and i32 %v265, 7
  %v275 = icmp eq i32 %v273, 0
  br i1 %v275, label %bb230, label %bb234

bb49:                                             ; preds = %bb235
  %v276 = load float, ptr addrspace(3) %v1435, align 4
  %v277 = fmul float %v1434, %v276
  %v278 = add i64 %v121, 1536
  %v279 = add i64 %v278, %v21
  %v280 = getelementptr inbounds i8, ptr %v12, i64 %v279
  %v281 = load i8, ptr %v280, align 1
  %v282 = zext i8 %v281 to i32
  %v283 = and i32 %v282, 128
  %v284 = trunc i64 24 to i32
  %v285 = and i32 %v284, 31
  %v286 = shl i32 %v283, %v285
  %v287 = and i32 %v282, 127
  %v288 = trunc i64 3 to i32
  %v289 = and i32 %v288, 31
  %v290 = lshr i32 %v287, %v289
  %v291 = and i32 %v282, 7
  %v292 = icmp eq i32 %v290, 0
  br i1 %v292, label %bb236, label %bb240

bb50:                                             ; preds = %bb241
  %v293 = load float, ptr addrspace(3) %v1453, align 4
  %v294 = fmul float %v1452, %v293
  %v295 = add i64 %v121, 1792
  %v296 = add i64 %v295, %v21
  %v297 = getelementptr inbounds i8, ptr %v12, i64 %v296
  %v298 = load i8, ptr %v297, align 1
  %v299 = zext i8 %v298 to i32
  %v300 = and i32 %v299, 128
  %v301 = trunc i64 24 to i32
  %v302 = and i32 %v301, 31
  %v303 = shl i32 %v300, %v302
  %v304 = and i32 %v299, 127
  %v305 = trunc i64 3 to i32
  %v306 = and i32 %v305, 31
  %v307 = lshr i32 %v304, %v306
  %v308 = and i32 %v299, 7
  %v309 = icmp eq i32 %v307, 0
  br i1 %v309, label %bb242, label %bb246

bb51:                                             ; preds = %bb247
  %v310 = load float, ptr addrspace(3) %v1471, align 4
  %v311 = fmul float %v1470, %v310
  %v312 = add i64 %v121, 2048
  %v313 = add i64 %v312, %v21
  %v314 = getelementptr inbounds i8, ptr %v12, i64 %v313
  %v315 = load i8, ptr %v314, align 1
  %v316 = zext i8 %v315 to i32
  %v317 = and i32 %v316, 128
  %v318 = trunc i64 24 to i32
  %v319 = and i32 %v318, 31
  %v320 = shl i32 %v317, %v319
  %v321 = and i32 %v316, 127
  %v322 = trunc i64 3 to i32
  %v323 = and i32 %v322, 31
  %v324 = lshr i32 %v321, %v323
  %v325 = and i32 %v316, 7
  %v326 = icmp eq i32 %v324, 0
  br i1 %v326, label %bb248, label %bb252

bb52:                                             ; preds = %bb253
  %v327 = load float, ptr addrspace(3) %v1489, align 4
  %v328 = fmul float %v1488, %v327
  %v329 = add i64 %v121, 2304
  %v330 = add i64 %v329, %v21
  %v331 = getelementptr inbounds i8, ptr %v12, i64 %v330
  %v332 = load i8, ptr %v331, align 1
  %v333 = zext i8 %v332 to i32
  %v334 = and i32 %v333, 128
  %v335 = trunc i64 24 to i32
  %v336 = and i32 %v335, 31
  %v337 = shl i32 %v334, %v336
  %v338 = and i32 %v333, 127
  %v339 = trunc i64 3 to i32
  %v340 = and i32 %v339, 31
  %v341 = lshr i32 %v338, %v340
  %v342 = and i32 %v333, 7
  %v343 = icmp eq i32 %v341, 0
  br i1 %v343, label %bb254, label %bb258

bb53:                                             ; preds = %bb259
  %v344 = load float, ptr addrspace(3) %v1507, align 4
  %v345 = fmul float %v1506, %v344
  %v346 = add i64 %v121, 2560
  %v347 = add i64 %v346, %v21
  %v348 = getelementptr inbounds i8, ptr %v12, i64 %v347
  %v349 = load i8, ptr %v348, align 1
  %v350 = zext i8 %v349 to i32
  %v351 = and i32 %v350, 128
  %v352 = trunc i64 24 to i32
  %v353 = and i32 %v352, 31
  %v354 = shl i32 %v351, %v353
  %v355 = and i32 %v350, 127
  %v356 = trunc i64 3 to i32
  %v357 = and i32 %v356, 31
  %v358 = lshr i32 %v355, %v357
  %v359 = and i32 %v350, 7
  %v360 = icmp eq i32 %v358, 0
  br i1 %v360, label %bb260, label %bb264

bb54:                                             ; preds = %bb265
  %v361 = load float, ptr addrspace(3) %v1525, align 4
  %v362 = fmul float %v1524, %v361
  %v363 = add i64 %v121, 2816
  %v364 = add i64 %v363, %v21
  %v365 = getelementptr inbounds i8, ptr %v12, i64 %v364
  %v366 = load i8, ptr %v365, align 1
  %v367 = zext i8 %v366 to i32
  %v368 = and i32 %v367, 128
  %v369 = trunc i64 24 to i32
  %v370 = and i32 %v369, 31
  %v371 = shl i32 %v368, %v370
  %v372 = and i32 %v367, 127
  %v373 = trunc i64 3 to i32
  %v374 = and i32 %v373, 31
  %v375 = lshr i32 %v372, %v374
  %v376 = and i32 %v367, 7
  %v377 = icmp eq i32 %v375, 0
  br i1 %v377, label %bb266, label %bb270

bb55:                                             ; preds = %bb271
  %v378 = load float, ptr addrspace(3) %v1543, align 4
  %v379 = fmul float %v1542, %v378
  %v380 = add i64 %v121, 3072
  %v381 = add i64 %v380, %v21
  %v382 = getelementptr inbounds i8, ptr %v12, i64 %v381
  %v383 = load i8, ptr %v382, align 1
  %v384 = zext i8 %v383 to i32
  %v385 = and i32 %v384, 128
  %v386 = trunc i64 24 to i32
  %v387 = and i32 %v386, 31
  %v388 = shl i32 %v385, %v387
  %v389 = and i32 %v384, 127
  %v390 = trunc i64 3 to i32
  %v391 = and i32 %v390, 31
  %v392 = lshr i32 %v389, %v391
  %v393 = and i32 %v384, 7
  %v394 = icmp eq i32 %v392, 0
  br i1 %v394, label %bb272, label %bb276

bb56:                                             ; preds = %bb277
  %v395 = load float, ptr addrspace(3) %v1561, align 4
  %v396 = fmul float %v1560, %v395
  %v397 = add i64 %v121, 3328
  %v398 = add i64 %v397, %v21
  %v399 = getelementptr inbounds i8, ptr %v12, i64 %v398
  %v400 = load i8, ptr %v399, align 1
  %v401 = zext i8 %v400 to i32
  %v402 = and i32 %v401, 128
  %v403 = trunc i64 24 to i32
  %v404 = and i32 %v403, 31
  %v405 = shl i32 %v402, %v404
  %v406 = and i32 %v401, 127
  %v407 = trunc i64 3 to i32
  %v408 = and i32 %v407, 31
  %v409 = lshr i32 %v406, %v408
  %v410 = and i32 %v401, 7
  %v411 = icmp eq i32 %v409, 0
  br i1 %v411, label %bb278, label %bb282

bb57:                                             ; preds = %bb283
  %v412 = load float, ptr addrspace(3) %v1579, align 4
  %v413 = fmul float %v1578, %v412
  %v414 = add i64 %v121, 3584
  %v415 = add i64 %v414, %v21
  %v416 = getelementptr inbounds i8, ptr %v12, i64 %v415
  %v417 = load i8, ptr %v416, align 1
  %v418 = zext i8 %v417 to i32
  %v419 = and i32 %v418, 128
  %v420 = trunc i64 24 to i32
  %v421 = and i32 %v420, 31
  %v422 = shl i32 %v419, %v421
  %v423 = and i32 %v418, 127
  %v424 = trunc i64 3 to i32
  %v425 = and i32 %v424, 31
  %v426 = lshr i32 %v423, %v425
  %v427 = and i32 %v418, 7
  %v428 = icmp eq i32 %v426, 0
  br i1 %v428, label %bb284, label %bb288

bb58:                                             ; preds = %bb289
  %v429 = load float, ptr addrspace(3) %v1597, align 4
  %v430 = fmul float %v1596, %v429
  %v431 = add i64 %v121, 3840
  %v432 = add i64 %v431, %v21
  %v433 = getelementptr inbounds i8, ptr %v12, i64 %v432
  %v434 = load i8, ptr %v433, align 1
  %v435 = zext i8 %v434 to i32
  %v436 = and i32 %v435, 128
  %v437 = trunc i64 24 to i32
  %v438 = and i32 %v437, 31
  %v439 = shl i32 %v436, %v438
  %v440 = and i32 %v435, 127
  %v441 = trunc i64 3 to i32
  %v442 = and i32 %v441, 31
  %v443 = lshr i32 %v440, %v442
  %v444 = and i32 %v435, 7
  %v445 = icmp eq i32 %v443, 0
  br i1 %v445, label %bb290, label %bb294

bb59:                                             ; preds = %bb295
  %v446 = load float, ptr addrspace(3) %v1615, align 4
  %v447 = fmul float %v1614, %v446
  %v448 = add i64 %v121, 4096
  %v449 = add i64 %v448, %v21
  %v450 = getelementptr inbounds i8, ptr %v12, i64 %v449
  %v451 = load i8, ptr %v450, align 1
  %v452 = zext i8 %v451 to i32
  %v453 = and i32 %v452, 128
  %v454 = trunc i64 24 to i32
  %v455 = and i32 %v454, 31
  %v456 = shl i32 %v453, %v455
  %v457 = and i32 %v452, 127
  %v458 = trunc i64 3 to i32
  %v459 = and i32 %v458, 31
  %v460 = lshr i32 %v457, %v459
  %v461 = and i32 %v452, 7
  %v462 = icmp eq i32 %v460, 0
  br i1 %v462, label %bb296, label %bb300

bb60:                                             ; preds = %bb301
  %v463 = load float, ptr addrspace(3) %v1633, align 4
  %v464 = fmul float %v1632, %v463
  %v465 = add i64 %v121, 4352
  %v466 = add i64 %v465, %v21
  %v467 = getelementptr inbounds i8, ptr %v12, i64 %v466
  %v468 = load i8, ptr %v467, align 1
  %v469 = zext i8 %v468 to i32
  %v470 = and i32 %v469, 128
  %v471 = trunc i64 24 to i32
  %v472 = and i32 %v471, 31
  %v473 = shl i32 %v470, %v472
  %v474 = and i32 %v469, 127
  %v475 = trunc i64 3 to i32
  %v476 = and i32 %v475, 31
  %v477 = lshr i32 %v474, %v476
  %v478 = and i32 %v469, 7
  %v479 = icmp eq i32 %v477, 0
  br i1 %v479, label %bb302, label %bb306

bb61:                                             ; preds = %bb307
  %v480 = load float, ptr addrspace(3) %v1651, align 4
  %v481 = fmul float %v1650, %v480
  %v482 = add i64 %v121, 4608
  %v483 = add i64 %v482, %v21
  %v484 = getelementptr inbounds i8, ptr %v12, i64 %v483
  %v485 = load i8, ptr %v484, align 1
  %v486 = zext i8 %v485 to i32
  %v487 = and i32 %v486, 128
  %v488 = trunc i64 24 to i32
  %v489 = and i32 %v488, 31
  %v490 = shl i32 %v487, %v489
  %v491 = and i32 %v486, 127
  %v492 = trunc i64 3 to i32
  %v493 = and i32 %v492, 31
  %v494 = lshr i32 %v491, %v493
  %v495 = and i32 %v486, 7
  %v496 = icmp eq i32 %v494, 0
  br i1 %v496, label %bb308, label %bb312

bb62:                                             ; preds = %bb313
  %v497 = load float, ptr addrspace(3) %v1669, align 4
  %v498 = fmul float %v1668, %v497
  %v499 = add i64 %v121, 4864
  %v500 = add i64 %v499, %v21
  %v501 = getelementptr inbounds i8, ptr %v12, i64 %v500
  %v502 = load i8, ptr %v501, align 1
  %v503 = zext i8 %v502 to i32
  %v504 = and i32 %v503, 128
  %v505 = trunc i64 24 to i32
  %v506 = and i32 %v505, 31
  %v507 = shl i32 %v504, %v506
  %v508 = and i32 %v503, 127
  %v509 = trunc i64 3 to i32
  %v510 = and i32 %v509, 31
  %v511 = lshr i32 %v508, %v510
  %v512 = and i32 %v503, 7
  %v513 = icmp eq i32 %v511, 0
  br i1 %v513, label %bb314, label %bb318

bb63:                                             ; preds = %bb319
  %v514 = load float, ptr addrspace(3) %v1687, align 4
  %v515 = fmul float %v1686, %v514
  %v516 = add i64 %v121, 5120
  %v517 = add i64 %v516, %v21
  %v518 = getelementptr inbounds i8, ptr %v12, i64 %v517
  %v519 = load i8, ptr %v518, align 1
  %v520 = zext i8 %v519 to i32
  %v521 = and i32 %v520, 128
  %v522 = trunc i64 24 to i32
  %v523 = and i32 %v522, 31
  %v524 = shl i32 %v521, %v523
  %v525 = and i32 %v520, 127
  %v526 = trunc i64 3 to i32
  %v527 = and i32 %v526, 31
  %v528 = lshr i32 %v525, %v527
  %v529 = and i32 %v520, 7
  %v530 = icmp eq i32 %v528, 0
  br i1 %v530, label %bb320, label %bb324

bb64:                                             ; preds = %bb325
  %v531 = load float, ptr addrspace(3) %v1705, align 4
  %v532 = fmul float %v1704, %v531
  %v533 = add i64 %v121, 5376
  %v534 = add i64 %v533, %v21
  %v535 = getelementptr inbounds i8, ptr %v12, i64 %v534
  %v536 = load i8, ptr %v535, align 1
  %v537 = zext i8 %v536 to i32
  %v538 = and i32 %v537, 128
  %v539 = trunc i64 24 to i32
  %v540 = and i32 %v539, 31
  %v541 = shl i32 %v538, %v540
  %v542 = and i32 %v537, 127
  %v543 = trunc i64 3 to i32
  %v544 = and i32 %v543, 31
  %v545 = lshr i32 %v542, %v544
  %v546 = and i32 %v537, 7
  %v547 = icmp eq i32 %v545, 0
  br i1 %v547, label %bb326, label %bb330

bb65:                                             ; preds = %bb331
  %v548 = load float, ptr addrspace(3) %v1723, align 4
  %v549 = fmul float %v1722, %v548
  %v550 = add i64 %v121, 5632
  %v551 = add i64 %v550, %v21
  %v552 = getelementptr inbounds i8, ptr %v12, i64 %v551
  %v553 = load i8, ptr %v552, align 1
  %v554 = zext i8 %v553 to i32
  %v555 = and i32 %v554, 128
  %v556 = trunc i64 24 to i32
  %v557 = and i32 %v556, 31
  %v558 = shl i32 %v555, %v557
  %v559 = and i32 %v554, 127
  %v560 = trunc i64 3 to i32
  %v561 = and i32 %v560, 31
  %v562 = lshr i32 %v559, %v561
  %v563 = and i32 %v554, 7
  %v564 = icmp eq i32 %v562, 0
  br i1 %v564, label %bb332, label %bb336

bb66:                                             ; preds = %bb337
  %v565 = load float, ptr addrspace(3) %v1741, align 4
  %v566 = fmul float %v1740, %v565
  %v567 = add i64 %v121, 5888
  %v568 = add i64 %v567, %v21
  %v569 = getelementptr inbounds i8, ptr %v12, i64 %v568
  %v570 = load i8, ptr %v569, align 1
  %v571 = zext i8 %v570 to i32
  %v572 = and i32 %v571, 128
  %v573 = trunc i64 24 to i32
  %v574 = and i32 %v573, 31
  %v575 = shl i32 %v572, %v574
  %v576 = and i32 %v571, 127
  %v577 = trunc i64 3 to i32
  %v578 = and i32 %v577, 31
  %v579 = lshr i32 %v576, %v578
  %v580 = and i32 %v571, 7
  %v581 = icmp eq i32 %v579, 0
  br i1 %v581, label %bb338, label %bb342

bb67:                                             ; preds = %bb343
  %v582 = load float, ptr addrspace(3) %v1759, align 4
  %v583 = fmul float %v1758, %v582
  %v584 = add i64 %v121, 6144
  %v585 = add i64 %v584, %v21
  %v586 = getelementptr inbounds i8, ptr %v12, i64 %v585
  %v587 = load i8, ptr %v586, align 1
  %v588 = zext i8 %v587 to i32
  %v589 = and i32 %v588, 128
  %v590 = trunc i64 24 to i32
  %v591 = and i32 %v590, 31
  %v592 = shl i32 %v589, %v591
  %v593 = and i32 %v588, 127
  %v594 = trunc i64 3 to i32
  %v595 = and i32 %v594, 31
  %v596 = lshr i32 %v593, %v595
  %v597 = and i32 %v588, 7
  %v598 = icmp eq i32 %v596, 0
  br i1 %v598, label %bb344, label %bb348

bb68:                                             ; preds = %bb349
  %v599 = load float, ptr addrspace(3) %v1777, align 4
  %v600 = fmul float %v1776, %v599
  %v601 = add i64 %v121, 6400
  %v602 = add i64 %v601, %v21
  %v603 = getelementptr inbounds i8, ptr %v12, i64 %v602
  %v604 = load i8, ptr %v603, align 1
  %v605 = zext i8 %v604 to i32
  %v606 = and i32 %v605, 128
  %v607 = trunc i64 24 to i32
  %v608 = and i32 %v607, 31
  %v609 = shl i32 %v606, %v608
  %v610 = and i32 %v605, 127
  %v611 = trunc i64 3 to i32
  %v612 = and i32 %v611, 31
  %v613 = lshr i32 %v610, %v612
  %v614 = and i32 %v605, 7
  %v615 = icmp eq i32 %v613, 0
  br i1 %v615, label %bb350, label %bb354

bb69:                                             ; preds = %bb355
  %v616 = load float, ptr addrspace(3) %v1795, align 4
  %v617 = fmul float %v1794, %v616
  %v618 = add i64 %v121, 6656
  %v619 = add i64 %v618, %v21
  %v620 = getelementptr inbounds i8, ptr %v12, i64 %v619
  %v621 = load i8, ptr %v620, align 1
  %v622 = zext i8 %v621 to i32
  %v623 = and i32 %v622, 128
  %v624 = trunc i64 24 to i32
  %v625 = and i32 %v624, 31
  %v626 = shl i32 %v623, %v625
  %v627 = and i32 %v622, 127
  %v628 = trunc i64 3 to i32
  %v629 = and i32 %v628, 31
  %v630 = lshr i32 %v627, %v629
  %v631 = and i32 %v622, 7
  %v632 = icmp eq i32 %v630, 0
  br i1 %v632, label %bb356, label %bb360

bb70:                                             ; preds = %bb361
  %v633 = load float, ptr addrspace(3) %v1813, align 4
  %v634 = fmul float %v1812, %v633
  %v635 = add i64 %v121, 6912
  %v636 = add i64 %v635, %v21
  %v637 = getelementptr inbounds i8, ptr %v12, i64 %v636
  %v638 = load i8, ptr %v637, align 1
  %v639 = zext i8 %v638 to i32
  %v640 = and i32 %v639, 128
  %v641 = trunc i64 24 to i32
  %v642 = and i32 %v641, 31
  %v643 = shl i32 %v640, %v642
  %v644 = and i32 %v639, 127
  %v645 = trunc i64 3 to i32
  %v646 = and i32 %v645, 31
  %v647 = lshr i32 %v644, %v646
  %v648 = and i32 %v639, 7
  %v649 = icmp eq i32 %v647, 0
  br i1 %v649, label %bb362, label %bb366

bb71:                                             ; preds = %bb367
  %v650 = load float, ptr addrspace(3) %v1831, align 4
  %v651 = fmul float %v1830, %v650
  %v652 = add i64 %v121, 7168
  %v653 = add i64 %v652, %v21
  %v654 = getelementptr inbounds i8, ptr %v12, i64 %v653
  %v655 = load i8, ptr %v654, align 1
  %v656 = zext i8 %v655 to i32
  %v657 = and i32 %v656, 128
  %v658 = trunc i64 24 to i32
  %v659 = and i32 %v658, 31
  %v660 = shl i32 %v657, %v659
  %v661 = and i32 %v656, 127
  %v662 = trunc i64 3 to i32
  %v663 = and i32 %v662, 31
  %v664 = lshr i32 %v661, %v663
  %v665 = and i32 %v656, 7
  %v666 = icmp eq i32 %v664, 0
  br i1 %v666, label %bb368, label %bb372

bb72:                                             ; preds = %bb373
  %v667 = load float, ptr addrspace(3) %v1849, align 4
  %v668 = fmul float %v1848, %v667
  %v669 = add i64 %v121, 7424
  %v670 = add i64 %v669, %v21
  %v671 = getelementptr inbounds i8, ptr %v12, i64 %v670
  %v672 = load i8, ptr %v671, align 1
  %v673 = zext i8 %v672 to i32
  %v674 = and i32 %v673, 128
  %v675 = trunc i64 24 to i32
  %v676 = and i32 %v675, 31
  %v677 = shl i32 %v674, %v676
  %v678 = and i32 %v673, 127
  %v679 = trunc i64 3 to i32
  %v680 = and i32 %v679, 31
  %v681 = lshr i32 %v678, %v680
  %v682 = and i32 %v673, 7
  %v683 = icmp eq i32 %v681, 0
  br i1 %v683, label %bb374, label %bb378

bb73:                                             ; preds = %bb379
  %v684 = load float, ptr addrspace(3) %v1867, align 4
  %v685 = fmul float %v1866, %v684
  %v686 = add i64 %v121, 7680
  %v687 = add i64 %v686, %v21
  %v688 = getelementptr inbounds i8, ptr %v12, i64 %v687
  %v689 = load i8, ptr %v688, align 1
  %v690 = zext i8 %v689 to i32
  %v691 = and i32 %v690, 128
  %v692 = trunc i64 24 to i32
  %v693 = and i32 %v692, 31
  %v694 = shl i32 %v691, %v693
  %v695 = and i32 %v690, 127
  %v696 = trunc i64 3 to i32
  %v697 = and i32 %v696, 31
  %v698 = lshr i32 %v695, %v697
  %v699 = and i32 %v690, 7
  %v700 = icmp eq i32 %v698, 0
  br i1 %v700, label %bb380, label %bb384

bb74:                                             ; preds = %bb385
  %v701 = load float, ptr addrspace(3) %v1885, align 4
  %v702 = fmul float %v1884, %v701
  %v703 = add i64 %v121, 7936
  %v704 = add i64 %v703, %v21
  %v705 = getelementptr inbounds i8, ptr %v12, i64 %v704
  %v706 = load i8, ptr %v705, align 1
  %v707 = zext i8 %v706 to i32
  %v708 = and i32 %v707, 128
  %v709 = trunc i64 24 to i32
  %v710 = and i32 %v709, 31
  %v711 = shl i32 %v708, %v710
  %v712 = and i32 %v707, 127
  %v713 = trunc i64 3 to i32
  %v714 = and i32 %v713, 31
  %v715 = lshr i32 %v712, %v714
  %v716 = and i32 %v707, 7
  %v717 = icmp eq i32 %v715, 0
  br i1 %v717, label %bb386, label %bb390

bb75:                                             ; preds = %bb391
  %v718 = load float, ptr addrspace(3) %v1903, align 4
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
  %v751 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_0, i64 2
  br label %bb76

bb76:                                             ; preds = %bb75
  %v752 = load float, ptr addrspace(3) %v751, align 4
  %v753 = fmul float %v106, %v752
  %v754 = fadd float %v753, %v750
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb77

bb77:                                             ; preds = %bb76
  %v756 = add i32 %v107, 1
  br label %bb15

bb78:                                             ; preds = %bb15
  %v757 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_0, i64 1
  br label %bb79

bb79:                                             ; preds = %bb78
  %v758 = load float, ptr addrspace(3) %v757, align 4
  %v759 = add i64 %v51, %v21
  %v760 = getelementptr inbounds float, ptr %v15, i64 %v759
  %v761 = fdiv float %v106, %v758
  store float %v761, ptr %v760, align 4
  br label %bb80

bb80:                                             ; preds = %bb79, %bb4
  ret void

bb81:                                             ; preds = %bb18
  %v762 = uitofp i32 %v142 to float
  %v763 = fmul float %v762, f0x3B000000
  %v764 = icmp eq i32 %v137, 0
  br i1 %v764, label %bb82, label %bb83

bb82:                                             ; preds = %bb81
  br label %bb84

bb83:                                             ; preds = %bb81
  %v765 = fneg float %v763
  br label %bb84

bb84:                                             ; preds = %bb83, %bb82
  %v766 = phi float [ %v763, %bb82 ], [ %v765, %bb83 ]
  br label %bb86

bb85:                                             ; preds = %bb18
  %v767 = add i32 %v141, 120
  %v768 = trunc i64 23 to i32
  %v769 = and i32 %v768, 31
  %v770 = shl i32 %v767, %v769
  %v771 = or i32 %v137, %v770
  %v772 = trunc i64 20 to i32
  %v773 = and i32 %v772, 31
  %v774 = shl i32 %v142, %v773
  %v775 = or i32 %v771, %v774
  %v776 = bitcast i32 %v775 to float
  br label %bb86

bb86:                                             ; preds = %bb85, %bb84
  %v777 = phi float [ %v766, %bb84 ], [ %v776, %bb85 ]
  %v778 = fmul float %v777, %v18
  %v779 = fmul float %v55, %v778
  %v780 = add i64 %v130, 1
  %v781 = getelementptr inbounds i8, ptr %v11, i64 %v780
  %v782 = load i8, ptr %v781, align 1
  %v783 = zext i8 %v782 to i32
  %v784 = and i32 %v783, 128
  %v785 = trunc i64 24 to i32
  %v786 = and i32 %v785, 31
  %v787 = shl i32 %v784, %v786
  %v788 = and i32 %v783, 127
  %v789 = trunc i64 3 to i32
  %v790 = and i32 %v789, 31
  %v791 = lshr i32 %v788, %v790
  %v792 = and i32 %v783, 7
  %v793 = icmp eq i32 %v791, 0
  br i1 %v793, label %bb87, label %bb91

bb87:                                             ; preds = %bb86
  %v794 = uitofp i32 %v792 to float
  %v795 = fmul float %v794, f0x3B000000
  %v796 = icmp eq i32 %v787, 0
  br i1 %v796, label %bb88, label %bb89

bb88:                                             ; preds = %bb87
  br label %bb90

bb89:                                             ; preds = %bb87
  %v797 = fneg float %v795
  br label %bb90

bb90:                                             ; preds = %bb89, %bb88
  %v798 = phi float [ %v795, %bb88 ], [ %v797, %bb89 ]
  br label %bb92

bb91:                                             ; preds = %bb86
  %v799 = add i32 %v791, 120
  %v800 = trunc i64 23 to i32
  %v801 = and i32 %v800, 31
  %v802 = shl i32 %v799, %v801
  %v803 = or i32 %v787, %v802
  %v804 = trunc i64 20 to i32
  %v805 = and i32 %v804, 31
  %v806 = shl i32 %v792, %v805
  %v807 = or i32 %v803, %v806
  %v808 = bitcast i32 %v807 to float
  br label %bb92

bb92:                                             ; preds = %bb91, %bb90
  %v809 = phi float [ %v798, %bb90 ], [ %v808, %bb91 ]
  %v810 = fmul float %v809, %v18
  %v811 = fmul float %v58, %v810
  %v812 = add i64 %v130, 2
  %v813 = getelementptr inbounds i8, ptr %v11, i64 %v812
  %v814 = load i8, ptr %v813, align 1
  %v815 = zext i8 %v814 to i32
  %v816 = and i32 %v815, 128
  %v817 = trunc i64 24 to i32
  %v818 = and i32 %v817, 31
  %v819 = shl i32 %v816, %v818
  %v820 = and i32 %v815, 127
  %v821 = trunc i64 3 to i32
  %v822 = and i32 %v821, 31
  %v823 = lshr i32 %v820, %v822
  %v824 = and i32 %v815, 7
  %v825 = icmp eq i32 %v823, 0
  br i1 %v825, label %bb93, label %bb97

bb93:                                             ; preds = %bb92
  %v826 = uitofp i32 %v824 to float
  %v827 = fmul float %v826, f0x3B000000
  %v828 = icmp eq i32 %v819, 0
  br i1 %v828, label %bb94, label %bb95

bb94:                                             ; preds = %bb93
  br label %bb96

bb95:                                             ; preds = %bb93
  %v829 = fneg float %v827
  br label %bb96

bb96:                                             ; preds = %bb95, %bb94
  %v830 = phi float [ %v827, %bb94 ], [ %v829, %bb95 ]
  br label %bb98

bb97:                                             ; preds = %bb92
  %v831 = add i32 %v823, 120
  %v832 = trunc i64 23 to i32
  %v833 = and i32 %v832, 31
  %v834 = shl i32 %v831, %v833
  %v835 = or i32 %v819, %v834
  %v836 = trunc i64 20 to i32
  %v837 = and i32 %v836, 31
  %v838 = shl i32 %v824, %v837
  %v839 = or i32 %v835, %v838
  %v840 = bitcast i32 %v839 to float
  br label %bb98

bb98:                                             ; preds = %bb97, %bb96
  %v841 = phi float [ %v830, %bb96 ], [ %v840, %bb97 ]
  %v842 = fmul float %v841, %v18
  %v843 = fmul float %v61, %v842
  %v844 = add i64 %v130, 3
  %v845 = getelementptr inbounds i8, ptr %v11, i64 %v844
  %v846 = load i8, ptr %v845, align 1
  %v847 = zext i8 %v846 to i32
  %v848 = and i32 %v847, 128
  %v849 = trunc i64 24 to i32
  %v850 = and i32 %v849, 31
  %v851 = shl i32 %v848, %v850
  %v852 = and i32 %v847, 127
  %v853 = trunc i64 3 to i32
  %v854 = and i32 %v853, 31
  %v855 = lshr i32 %v852, %v854
  %v856 = and i32 %v847, 7
  %v857 = icmp eq i32 %v855, 0
  br i1 %v857, label %bb99, label %bb103

bb99:                                             ; preds = %bb98
  %v858 = uitofp i32 %v856 to float
  %v859 = fmul float %v858, f0x3B000000
  %v860 = icmp eq i32 %v851, 0
  br i1 %v860, label %bb100, label %bb101

bb100:                                            ; preds = %bb99
  br label %bb102

bb101:                                            ; preds = %bb99
  %v861 = fneg float %v859
  br label %bb102

bb102:                                            ; preds = %bb101, %bb100
  %v862 = phi float [ %v859, %bb100 ], [ %v861, %bb101 ]
  br label %bb104

bb103:                                            ; preds = %bb98
  %v863 = add i32 %v855, 120
  %v864 = trunc i64 23 to i32
  %v865 = and i32 %v864, 31
  %v866 = shl i32 %v863, %v865
  %v867 = or i32 %v851, %v866
  %v868 = trunc i64 20 to i32
  %v869 = and i32 %v868, 31
  %v870 = shl i32 %v856, %v869
  %v871 = or i32 %v867, %v870
  %v872 = bitcast i32 %v871 to float
  br label %bb104

bb104:                                            ; preds = %bb103, %bb102
  %v873 = phi float [ %v862, %bb102 ], [ %v872, %bb103 ]
  %v874 = fmul float %v873, %v18
  %v875 = fmul float %v64, %v874
  %v876 = add i64 %v130, 4
  %v877 = getelementptr inbounds i8, ptr %v11, i64 %v876
  %v878 = load i8, ptr %v877, align 1
  %v879 = zext i8 %v878 to i32
  %v880 = and i32 %v879, 128
  %v881 = trunc i64 24 to i32
  %v882 = and i32 %v881, 31
  %v883 = shl i32 %v880, %v882
  %v884 = and i32 %v879, 127
  %v885 = trunc i64 3 to i32
  %v886 = and i32 %v885, 31
  %v887 = lshr i32 %v884, %v886
  %v888 = and i32 %v879, 7
  %v889 = icmp eq i32 %v887, 0
  br i1 %v889, label %bb105, label %bb109

bb105:                                            ; preds = %bb104
  %v890 = uitofp i32 %v888 to float
  %v891 = fmul float %v890, f0x3B000000
  %v892 = icmp eq i32 %v883, 0
  br i1 %v892, label %bb106, label %bb107

bb106:                                            ; preds = %bb105
  br label %bb108

bb107:                                            ; preds = %bb105
  %v893 = fneg float %v891
  br label %bb108

bb108:                                            ; preds = %bb107, %bb106
  %v894 = phi float [ %v891, %bb106 ], [ %v893, %bb107 ]
  br label %bb110

bb109:                                            ; preds = %bb104
  %v895 = add i32 %v887, 120
  %v896 = trunc i64 23 to i32
  %v897 = and i32 %v896, 31
  %v898 = shl i32 %v895, %v897
  %v899 = or i32 %v883, %v898
  %v900 = trunc i64 20 to i32
  %v901 = and i32 %v900, 31
  %v902 = shl i32 %v888, %v901
  %v903 = or i32 %v899, %v902
  %v904 = bitcast i32 %v903 to float
  br label %bb110

bb110:                                            ; preds = %bb109, %bb108
  %v905 = phi float [ %v894, %bb108 ], [ %v904, %bb109 ]
  %v906 = fmul float %v905, %v18
  %v907 = fmul float %v67, %v906
  %v908 = add i64 %v130, 5
  %v909 = getelementptr inbounds i8, ptr %v11, i64 %v908
  %v910 = load i8, ptr %v909, align 1
  %v911 = zext i8 %v910 to i32
  %v912 = and i32 %v911, 128
  %v913 = trunc i64 24 to i32
  %v914 = and i32 %v913, 31
  %v915 = shl i32 %v912, %v914
  %v916 = and i32 %v911, 127
  %v917 = trunc i64 3 to i32
  %v918 = and i32 %v917, 31
  %v919 = lshr i32 %v916, %v918
  %v920 = and i32 %v911, 7
  %v921 = icmp eq i32 %v919, 0
  br i1 %v921, label %bb111, label %bb115

bb111:                                            ; preds = %bb110
  %v922 = uitofp i32 %v920 to float
  %v923 = fmul float %v922, f0x3B000000
  %v924 = icmp eq i32 %v915, 0
  br i1 %v924, label %bb112, label %bb113

bb112:                                            ; preds = %bb111
  br label %bb114

bb113:                                            ; preds = %bb111
  %v925 = fneg float %v923
  br label %bb114

bb114:                                            ; preds = %bb113, %bb112
  %v926 = phi float [ %v923, %bb112 ], [ %v925, %bb113 ]
  br label %bb116

bb115:                                            ; preds = %bb110
  %v927 = add i32 %v919, 120
  %v928 = trunc i64 23 to i32
  %v929 = and i32 %v928, 31
  %v930 = shl i32 %v927, %v929
  %v931 = or i32 %v915, %v930
  %v932 = trunc i64 20 to i32
  %v933 = and i32 %v932, 31
  %v934 = shl i32 %v920, %v933
  %v935 = or i32 %v931, %v934
  %v936 = bitcast i32 %v935 to float
  br label %bb116

bb116:                                            ; preds = %bb115, %bb114
  %v937 = phi float [ %v926, %bb114 ], [ %v936, %bb115 ]
  %v938 = fmul float %v937, %v18
  %v939 = fmul float %v70, %v938
  %v940 = add i64 %v130, 6
  %v941 = getelementptr inbounds i8, ptr %v11, i64 %v940
  %v942 = load i8, ptr %v941, align 1
  %v943 = zext i8 %v942 to i32
  %v944 = and i32 %v943, 128
  %v945 = trunc i64 24 to i32
  %v946 = and i32 %v945, 31
  %v947 = shl i32 %v944, %v946
  %v948 = and i32 %v943, 127
  %v949 = trunc i64 3 to i32
  %v950 = and i32 %v949, 31
  %v951 = lshr i32 %v948, %v950
  %v952 = and i32 %v943, 7
  %v953 = icmp eq i32 %v951, 0
  br i1 %v953, label %bb117, label %bb121

bb117:                                            ; preds = %bb116
  %v954 = uitofp i32 %v952 to float
  %v955 = fmul float %v954, f0x3B000000
  %v956 = icmp eq i32 %v947, 0
  br i1 %v956, label %bb118, label %bb119

bb118:                                            ; preds = %bb117
  br label %bb120

bb119:                                            ; preds = %bb117
  %v957 = fneg float %v955
  br label %bb120

bb120:                                            ; preds = %bb119, %bb118
  %v958 = phi float [ %v955, %bb118 ], [ %v957, %bb119 ]
  br label %bb122

bb121:                                            ; preds = %bb116
  %v959 = add i32 %v951, 120
  %v960 = trunc i64 23 to i32
  %v961 = and i32 %v960, 31
  %v962 = shl i32 %v959, %v961
  %v963 = or i32 %v947, %v962
  %v964 = trunc i64 20 to i32
  %v965 = and i32 %v964, 31
  %v966 = shl i32 %v952, %v965
  %v967 = or i32 %v963, %v966
  %v968 = bitcast i32 %v967 to float
  br label %bb122

bb122:                                            ; preds = %bb121, %bb120
  %v969 = phi float [ %v958, %bb120 ], [ %v968, %bb121 ]
  %v970 = fmul float %v969, %v18
  %v971 = fmul float %v73, %v970
  %v972 = add i64 %v130, 7
  %v973 = getelementptr inbounds i8, ptr %v11, i64 %v972
  %v974 = load i8, ptr %v973, align 1
  %v975 = zext i8 %v974 to i32
  %v976 = and i32 %v975, 128
  %v977 = trunc i64 24 to i32
  %v978 = and i32 %v977, 31
  %v979 = shl i32 %v976, %v978
  %v980 = and i32 %v975, 127
  %v981 = trunc i64 3 to i32
  %v982 = and i32 %v981, 31
  %v983 = lshr i32 %v980, %v982
  %v984 = and i32 %v975, 7
  %v985 = icmp eq i32 %v983, 0
  br i1 %v985, label %bb123, label %bb127

bb123:                                            ; preds = %bb122
  %v986 = uitofp i32 %v984 to float
  %v987 = fmul float %v986, f0x3B000000
  %v988 = icmp eq i32 %v979, 0
  br i1 %v988, label %bb124, label %bb125

bb124:                                            ; preds = %bb123
  br label %bb126

bb125:                                            ; preds = %bb123
  %v989 = fneg float %v987
  br label %bb126

bb126:                                            ; preds = %bb125, %bb124
  %v990 = phi float [ %v987, %bb124 ], [ %v989, %bb125 ]
  br label %bb128

bb127:                                            ; preds = %bb122
  %v991 = add i32 %v983, 120
  %v992 = trunc i64 23 to i32
  %v993 = and i32 %v992, 31
  %v994 = shl i32 %v991, %v993
  %v995 = or i32 %v979, %v994
  %v996 = trunc i64 20 to i32
  %v997 = and i32 %v996, 31
  %v998 = shl i32 %v984, %v997
  %v999 = or i32 %v995, %v998
  %v1000 = bitcast i32 %v999 to float
  br label %bb128

bb128:                                            ; preds = %bb127, %bb126
  %v1001 = phi float [ %v990, %bb126 ], [ %v1000, %bb127 ]
  %v1002 = fmul float %v1001, %v18
  %v1003 = fmul float %v76, %v1002
  %v1004 = add i64 %v130, 8
  %v1005 = getelementptr inbounds i8, ptr %v11, i64 %v1004
  %v1006 = load i8, ptr %v1005, align 1
  %v1007 = zext i8 %v1006 to i32
  %v1008 = and i32 %v1007, 128
  %v1009 = trunc i64 24 to i32
  %v1010 = and i32 %v1009, 31
  %v1011 = shl i32 %v1008, %v1010
  %v1012 = and i32 %v1007, 127
  %v1013 = trunc i64 3 to i32
  %v1014 = and i32 %v1013, 31
  %v1015 = lshr i32 %v1012, %v1014
  %v1016 = and i32 %v1007, 7
  %v1017 = icmp eq i32 %v1015, 0
  br i1 %v1017, label %bb129, label %bb133

bb129:                                            ; preds = %bb128
  %v1018 = uitofp i32 %v1016 to float
  %v1019 = fmul float %v1018, f0x3B000000
  %v1020 = icmp eq i32 %v1011, 0
  br i1 %v1020, label %bb130, label %bb131

bb130:                                            ; preds = %bb129
  br label %bb132

bb131:                                            ; preds = %bb129
  %v1021 = fneg float %v1019
  br label %bb132

bb132:                                            ; preds = %bb131, %bb130
  %v1022 = phi float [ %v1019, %bb130 ], [ %v1021, %bb131 ]
  br label %bb134

bb133:                                            ; preds = %bb128
  %v1023 = add i32 %v1015, 120
  %v1024 = trunc i64 23 to i32
  %v1025 = and i32 %v1024, 31
  %v1026 = shl i32 %v1023, %v1025
  %v1027 = or i32 %v1011, %v1026
  %v1028 = trunc i64 20 to i32
  %v1029 = and i32 %v1028, 31
  %v1030 = shl i32 %v1016, %v1029
  %v1031 = or i32 %v1027, %v1030
  %v1032 = bitcast i32 %v1031 to float
  br label %bb134

bb134:                                            ; preds = %bb133, %bb132
  %v1033 = phi float [ %v1022, %bb132 ], [ %v1032, %bb133 ]
  %v1034 = fmul float %v1033, %v18
  %v1035 = fmul float %v79, %v1034
  %v1036 = add i64 %v130, 9
  %v1037 = getelementptr inbounds i8, ptr %v11, i64 %v1036
  %v1038 = load i8, ptr %v1037, align 1
  %v1039 = zext i8 %v1038 to i32
  %v1040 = and i32 %v1039, 128
  %v1041 = trunc i64 24 to i32
  %v1042 = and i32 %v1041, 31
  %v1043 = shl i32 %v1040, %v1042
  %v1044 = and i32 %v1039, 127
  %v1045 = trunc i64 3 to i32
  %v1046 = and i32 %v1045, 31
  %v1047 = lshr i32 %v1044, %v1046
  %v1048 = and i32 %v1039, 7
  %v1049 = icmp eq i32 %v1047, 0
  br i1 %v1049, label %bb135, label %bb139

bb135:                                            ; preds = %bb134
  %v1050 = uitofp i32 %v1048 to float
  %v1051 = fmul float %v1050, f0x3B000000
  %v1052 = icmp eq i32 %v1043, 0
  br i1 %v1052, label %bb136, label %bb137

bb136:                                            ; preds = %bb135
  br label %bb138

bb137:                                            ; preds = %bb135
  %v1053 = fneg float %v1051
  br label %bb138

bb138:                                            ; preds = %bb137, %bb136
  %v1054 = phi float [ %v1051, %bb136 ], [ %v1053, %bb137 ]
  br label %bb140

bb139:                                            ; preds = %bb134
  %v1055 = add i32 %v1047, 120
  %v1056 = trunc i64 23 to i32
  %v1057 = and i32 %v1056, 31
  %v1058 = shl i32 %v1055, %v1057
  %v1059 = or i32 %v1043, %v1058
  %v1060 = trunc i64 20 to i32
  %v1061 = and i32 %v1060, 31
  %v1062 = shl i32 %v1048, %v1061
  %v1063 = or i32 %v1059, %v1062
  %v1064 = bitcast i32 %v1063 to float
  br label %bb140

bb140:                                            ; preds = %bb139, %bb138
  %v1065 = phi float [ %v1054, %bb138 ], [ %v1064, %bb139 ]
  %v1066 = fmul float %v1065, %v18
  %v1067 = fmul float %v82, %v1066
  %v1068 = add i64 %v130, 10
  %v1069 = getelementptr inbounds i8, ptr %v11, i64 %v1068
  %v1070 = load i8, ptr %v1069, align 1
  %v1071 = zext i8 %v1070 to i32
  %v1072 = and i32 %v1071, 128
  %v1073 = trunc i64 24 to i32
  %v1074 = and i32 %v1073, 31
  %v1075 = shl i32 %v1072, %v1074
  %v1076 = and i32 %v1071, 127
  %v1077 = trunc i64 3 to i32
  %v1078 = and i32 %v1077, 31
  %v1079 = lshr i32 %v1076, %v1078
  %v1080 = and i32 %v1071, 7
  %v1081 = icmp eq i32 %v1079, 0
  br i1 %v1081, label %bb141, label %bb145

bb141:                                            ; preds = %bb140
  %v1082 = uitofp i32 %v1080 to float
  %v1083 = fmul float %v1082, f0x3B000000
  %v1084 = icmp eq i32 %v1075, 0
  br i1 %v1084, label %bb142, label %bb143

bb142:                                            ; preds = %bb141
  br label %bb144

bb143:                                            ; preds = %bb141
  %v1085 = fneg float %v1083
  br label %bb144

bb144:                                            ; preds = %bb143, %bb142
  %v1086 = phi float [ %v1083, %bb142 ], [ %v1085, %bb143 ]
  br label %bb146

bb145:                                            ; preds = %bb140
  %v1087 = add i32 %v1079, 120
  %v1088 = trunc i64 23 to i32
  %v1089 = and i32 %v1088, 31
  %v1090 = shl i32 %v1087, %v1089
  %v1091 = or i32 %v1075, %v1090
  %v1092 = trunc i64 20 to i32
  %v1093 = and i32 %v1092, 31
  %v1094 = shl i32 %v1080, %v1093
  %v1095 = or i32 %v1091, %v1094
  %v1096 = bitcast i32 %v1095 to float
  br label %bb146

bb146:                                            ; preds = %bb145, %bb144
  %v1097 = phi float [ %v1086, %bb144 ], [ %v1096, %bb145 ]
  %v1098 = fmul float %v1097, %v18
  %v1099 = fmul float %v85, %v1098
  %v1100 = add i64 %v130, 11
  %v1101 = getelementptr inbounds i8, ptr %v11, i64 %v1100
  %v1102 = load i8, ptr %v1101, align 1
  %v1103 = zext i8 %v1102 to i32
  %v1104 = and i32 %v1103, 128
  %v1105 = trunc i64 24 to i32
  %v1106 = and i32 %v1105, 31
  %v1107 = shl i32 %v1104, %v1106
  %v1108 = and i32 %v1103, 127
  %v1109 = trunc i64 3 to i32
  %v1110 = and i32 %v1109, 31
  %v1111 = lshr i32 %v1108, %v1110
  %v1112 = and i32 %v1103, 7
  %v1113 = icmp eq i32 %v1111, 0
  br i1 %v1113, label %bb147, label %bb151

bb147:                                            ; preds = %bb146
  %v1114 = uitofp i32 %v1112 to float
  %v1115 = fmul float %v1114, f0x3B000000
  %v1116 = icmp eq i32 %v1107, 0
  br i1 %v1116, label %bb148, label %bb149

bb148:                                            ; preds = %bb147
  br label %bb150

bb149:                                            ; preds = %bb147
  %v1117 = fneg float %v1115
  br label %bb150

bb150:                                            ; preds = %bb149, %bb148
  %v1118 = phi float [ %v1115, %bb148 ], [ %v1117, %bb149 ]
  br label %bb152

bb151:                                            ; preds = %bb146
  %v1119 = add i32 %v1111, 120
  %v1120 = trunc i64 23 to i32
  %v1121 = and i32 %v1120, 31
  %v1122 = shl i32 %v1119, %v1121
  %v1123 = or i32 %v1107, %v1122
  %v1124 = trunc i64 20 to i32
  %v1125 = and i32 %v1124, 31
  %v1126 = shl i32 %v1112, %v1125
  %v1127 = or i32 %v1123, %v1126
  %v1128 = bitcast i32 %v1127 to float
  br label %bb152

bb152:                                            ; preds = %bb151, %bb150
  %v1129 = phi float [ %v1118, %bb150 ], [ %v1128, %bb151 ]
  %v1130 = fmul float %v1129, %v18
  %v1131 = fmul float %v88, %v1130
  %v1132 = add i64 %v130, 12
  %v1133 = getelementptr inbounds i8, ptr %v11, i64 %v1132
  %v1134 = load i8, ptr %v1133, align 1
  %v1135 = zext i8 %v1134 to i32
  %v1136 = and i32 %v1135, 128
  %v1137 = trunc i64 24 to i32
  %v1138 = and i32 %v1137, 31
  %v1139 = shl i32 %v1136, %v1138
  %v1140 = and i32 %v1135, 127
  %v1141 = trunc i64 3 to i32
  %v1142 = and i32 %v1141, 31
  %v1143 = lshr i32 %v1140, %v1142
  %v1144 = and i32 %v1135, 7
  %v1145 = icmp eq i32 %v1143, 0
  br i1 %v1145, label %bb153, label %bb157

bb153:                                            ; preds = %bb152
  %v1146 = uitofp i32 %v1144 to float
  %v1147 = fmul float %v1146, f0x3B000000
  %v1148 = icmp eq i32 %v1139, 0
  br i1 %v1148, label %bb154, label %bb155

bb154:                                            ; preds = %bb153
  br label %bb156

bb155:                                            ; preds = %bb153
  %v1149 = fneg float %v1147
  br label %bb156

bb156:                                            ; preds = %bb155, %bb154
  %v1150 = phi float [ %v1147, %bb154 ], [ %v1149, %bb155 ]
  br label %bb158

bb157:                                            ; preds = %bb152
  %v1151 = add i32 %v1143, 120
  %v1152 = trunc i64 23 to i32
  %v1153 = and i32 %v1152, 31
  %v1154 = shl i32 %v1151, %v1153
  %v1155 = or i32 %v1139, %v1154
  %v1156 = trunc i64 20 to i32
  %v1157 = and i32 %v1156, 31
  %v1158 = shl i32 %v1144, %v1157
  %v1159 = or i32 %v1155, %v1158
  %v1160 = bitcast i32 %v1159 to float
  br label %bb158

bb158:                                            ; preds = %bb157, %bb156
  %v1161 = phi float [ %v1150, %bb156 ], [ %v1160, %bb157 ]
  %v1162 = fmul float %v1161, %v18
  %v1163 = fmul float %v91, %v1162
  %v1164 = add i64 %v130, 13
  %v1165 = getelementptr inbounds i8, ptr %v11, i64 %v1164
  %v1166 = load i8, ptr %v1165, align 1
  %v1167 = zext i8 %v1166 to i32
  %v1168 = and i32 %v1167, 128
  %v1169 = trunc i64 24 to i32
  %v1170 = and i32 %v1169, 31
  %v1171 = shl i32 %v1168, %v1170
  %v1172 = and i32 %v1167, 127
  %v1173 = trunc i64 3 to i32
  %v1174 = and i32 %v1173, 31
  %v1175 = lshr i32 %v1172, %v1174
  %v1176 = and i32 %v1167, 7
  %v1177 = icmp eq i32 %v1175, 0
  br i1 %v1177, label %bb159, label %bb163

bb159:                                            ; preds = %bb158
  %v1178 = uitofp i32 %v1176 to float
  %v1179 = fmul float %v1178, f0x3B000000
  %v1180 = icmp eq i32 %v1171, 0
  br i1 %v1180, label %bb160, label %bb161

bb160:                                            ; preds = %bb159
  br label %bb162

bb161:                                            ; preds = %bb159
  %v1181 = fneg float %v1179
  br label %bb162

bb162:                                            ; preds = %bb161, %bb160
  %v1182 = phi float [ %v1179, %bb160 ], [ %v1181, %bb161 ]
  br label %bb164

bb163:                                            ; preds = %bb158
  %v1183 = add i32 %v1175, 120
  %v1184 = trunc i64 23 to i32
  %v1185 = and i32 %v1184, 31
  %v1186 = shl i32 %v1183, %v1185
  %v1187 = or i32 %v1171, %v1186
  %v1188 = trunc i64 20 to i32
  %v1189 = and i32 %v1188, 31
  %v1190 = shl i32 %v1176, %v1189
  %v1191 = or i32 %v1187, %v1190
  %v1192 = bitcast i32 %v1191 to float
  br label %bb164

bb164:                                            ; preds = %bb163, %bb162
  %v1193 = phi float [ %v1182, %bb162 ], [ %v1192, %bb163 ]
  %v1194 = fmul float %v1193, %v18
  %v1195 = fmul float %v94, %v1194
  %v1196 = add i64 %v130, 14
  %v1197 = getelementptr inbounds i8, ptr %v11, i64 %v1196
  %v1198 = load i8, ptr %v1197, align 1
  %v1199 = zext i8 %v1198 to i32
  %v1200 = and i32 %v1199, 128
  %v1201 = trunc i64 24 to i32
  %v1202 = and i32 %v1201, 31
  %v1203 = shl i32 %v1200, %v1202
  %v1204 = and i32 %v1199, 127
  %v1205 = trunc i64 3 to i32
  %v1206 = and i32 %v1205, 31
  %v1207 = lshr i32 %v1204, %v1206
  %v1208 = and i32 %v1199, 7
  %v1209 = icmp eq i32 %v1207, 0
  br i1 %v1209, label %bb165, label %bb169

bb165:                                            ; preds = %bb164
  %v1210 = uitofp i32 %v1208 to float
  %v1211 = fmul float %v1210, f0x3B000000
  %v1212 = icmp eq i32 %v1203, 0
  br i1 %v1212, label %bb166, label %bb167

bb166:                                            ; preds = %bb165
  br label %bb168

bb167:                                            ; preds = %bb165
  %v1213 = fneg float %v1211
  br label %bb168

bb168:                                            ; preds = %bb167, %bb166
  %v1214 = phi float [ %v1211, %bb166 ], [ %v1213, %bb167 ]
  br label %bb170

bb169:                                            ; preds = %bb164
  %v1215 = add i32 %v1207, 120
  %v1216 = trunc i64 23 to i32
  %v1217 = and i32 %v1216, 31
  %v1218 = shl i32 %v1215, %v1217
  %v1219 = or i32 %v1203, %v1218
  %v1220 = trunc i64 20 to i32
  %v1221 = and i32 %v1220, 31
  %v1222 = shl i32 %v1208, %v1221
  %v1223 = or i32 %v1219, %v1222
  %v1224 = bitcast i32 %v1223 to float
  br label %bb170

bb170:                                            ; preds = %bb169, %bb168
  %v1225 = phi float [ %v1214, %bb168 ], [ %v1224, %bb169 ]
  %v1226 = fmul float %v1225, %v18
  %v1227 = fmul float %v97, %v1226
  %v1228 = add i64 %v130, 15
  %v1229 = getelementptr inbounds i8, ptr %v11, i64 %v1228
  %v1230 = load i8, ptr %v1229, align 1
  %v1231 = zext i8 %v1230 to i32
  %v1232 = and i32 %v1231, 128
  %v1233 = trunc i64 24 to i32
  %v1234 = and i32 %v1233, 31
  %v1235 = shl i32 %v1232, %v1234
  %v1236 = and i32 %v1231, 127
  %v1237 = trunc i64 3 to i32
  %v1238 = and i32 %v1237, 31
  %v1239 = lshr i32 %v1236, %v1238
  %v1240 = and i32 %v1231, 7
  %v1241 = icmp eq i32 %v1239, 0
  br i1 %v1241, label %bb171, label %bb175

bb171:                                            ; preds = %bb170
  %v1242 = uitofp i32 %v1240 to float
  %v1243 = fmul float %v1242, f0x3B000000
  %v1244 = icmp eq i32 %v1235, 0
  br i1 %v1244, label %bb172, label %bb173

bb172:                                            ; preds = %bb171
  br label %bb174

bb173:                                            ; preds = %bb171
  %v1245 = fneg float %v1243
  br label %bb174

bb174:                                            ; preds = %bb173, %bb172
  %v1246 = phi float [ %v1243, %bb172 ], [ %v1245, %bb173 ]
  br label %bb176

bb175:                                            ; preds = %bb170
  %v1247 = add i32 %v1239, 120
  %v1248 = trunc i64 23 to i32
  %v1249 = and i32 %v1248, 31
  %v1250 = shl i32 %v1247, %v1249
  %v1251 = or i32 %v1235, %v1250
  %v1252 = trunc i64 20 to i32
  %v1253 = and i32 %v1252, 31
  %v1254 = shl i32 %v1240, %v1253
  %v1255 = or i32 %v1251, %v1254
  %v1256 = bitcast i32 %v1255 to float
  br label %bb176

bb176:                                            ; preds = %bb175, %bb174
  %v1257 = phi float [ %v1246, %bb174 ], [ %v1256, %bb175 ]
  %v1258 = fmul float %v1257, %v18
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
  %v1275 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1274, i32 8, i32 31) #0
  br label %bb177

bb177:                                            ; preds = %bb176
  %v1276 = fadd float %v1274, %v1275
  %v1277 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1276, i32 4, i32 31) #0
  br label %bb178

bb178:                                            ; preds = %bb177
  %v1278 = fadd float %v1276, %v1277
  %v1279 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1278, i32 2, i32 31) #0
  br label %bb179

bb179:                                            ; preds = %bb178
  %v1280 = fadd float %v1278, %v1279
  %v1281 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1280, i32 1, i32 31) #0
  br label %bb180

bb180:                                            ; preds = %bb179
  %v1282 = fadd float %v1280, %v1281
  %v1283 = icmp eq i64 %v50, 0
  br i1 %v1283, label %bb19, label %bb26

bb181:                                            ; preds = %bb30
  %v1284 = fcmp uno float %v158, %v158
  %v1285 = fcmp oge float %v159, %v158
  %v1286 = select i1 %v1285, float %v159, float %v158
  %v1287 = select i1 %v1284, float %v159, float %v1286
  br label %bb182

bb182:                                            ; preds = %bb181
  %v1288 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1287, i32 2, i32 31) #0
  br label %bb183

bb183:                                            ; preds = %bb182
  %v1289 = fcmp uno float %v1287, %v1287
  %v1290 = fcmp oge float %v1288, %v1287
  %v1291 = select i1 %v1290, float %v1288, float %v1287
  %v1292 = select i1 %v1289, float %v1288, float %v1291
  br label %bb184

bb184:                                            ; preds = %bb183
  %v1293 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1292, i32 4, i32 31) #0
  br label %bb185

bb185:                                            ; preds = %bb184
  %v1294 = fcmp uno float %v1292, %v1292
  %v1295 = fcmp oge float %v1293, %v1292
  %v1296 = select i1 %v1295, float %v1293, float %v1292
  %v1297 = select i1 %v1294, float %v1293, float %v1296
  br label %bb186

bb186:                                            ; preds = %bb185
  %v1298 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1297, i32 8, i32 31) #0
  br label %bb187

bb187:                                            ; preds = %bb186
  %v1299 = fcmp uno float %v1297, %v1297
  %v1300 = fcmp oge float %v1298, %v1297
  %v1301 = select i1 %v1300, float %v1298, float %v1297
  %v1302 = select i1 %v1299, float %v1298, float %v1301
  br label %bb188

bb188:                                            ; preds = %bb187
  %v1303 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1302, i32 16, i32 31) #0
  br label %bb189

bb189:                                            ; preds = %bb188
  %v1304 = fcmp uno float %v1302, %v1302
  %v1305 = fcmp oge float %v1303, %v1302
  %v1306 = select i1 %v1305, float %v1303, float %v1302
  %v1307 = select i1 %v1304, float %v1303, float %v1306
  br label %bb190

bb190:                                            ; preds = %bb189
  %v1308 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_0, i64 0
  br label %bb31

bb191:                                            ; preds = %bb34
  %v1309 = fsub float %v158, %v163
  %v1310 = call float @__nv_expf(float %v1309) #0
  br label %bb192

bb192:                                            ; preds = %bb191
  %v1311 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 %v23
  br label %bb35

bb193:                                            ; preds = %bb35
  %v1312 = add i32 %v167, 8
  %v1313 = call float @llvm.nvvm.shfl.sync.idx.f32(i32 -1, float %v1310, i32 %v1312, i32 31) #0
  br label %bb194

bb194:                                            ; preds = %bb193
  %v1314 = fadd float %v168, %v1313
  %v1315 = add i32 %v167, 16
  %v1316 = call float @llvm.nvvm.shfl.sync.idx.f32(i32 -1, float %v1310, i32 %v1315, i32 31) #0
  br label %bb195

bb195:                                            ; preds = %bb194
  %v1317 = fadd float %v1314, %v1316
  %v1318 = add i32 %v167, 24
  %v1319 = call float @llvm.nvvm.shfl.sync.idx.f32(i32 -1, float %v1310, i32 %v1318, i32 31) #0
  br label %bb196

bb196:                                            ; preds = %bb195
  %v1320 = fadd float %v1317, %v1319
  %v1321 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1320, i32 1, i32 31) #0
  br label %bb197

bb197:                                            ; preds = %bb196
  %v1322 = fadd float %v1320, %v1321
  %v1323 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1322, i32 4, i32 31) #0
  br label %bb198

bb198:                                            ; preds = %bb197
  %v1324 = fadd float %v1322, %v1323
  %v1325 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %v1324, i32 2, i32 31) #0
  br label %bb199

bb199:                                            ; preds = %bb198
  %v1326 = fadd float %v1324, %v1325
  %v1327 = icmp eq i64 %v23, 0
  br i1 %v1327, label %bb36, label %bb41

bb200:                                            ; preds = %bb43
  %v1328 = uitofp i32 %v189 to float
  %v1329 = fmul float %v1328, f0x3B000000
  %v1330 = icmp eq i32 %v184, 0
  br i1 %v1330, label %bb201, label %bb202

bb201:                                            ; preds = %bb200
  br label %bb203

bb202:                                            ; preds = %bb200
  %v1331 = fneg float %v1329
  br label %bb203

bb203:                                            ; preds = %bb202, %bb201
  %v1332 = phi float [ %v1329, %bb201 ], [ %v1331, %bb202 ]
  br label %bb205

bb204:                                            ; preds = %bb43
  %v1333 = add i32 %v188, 120
  %v1334 = trunc i64 23 to i32
  %v1335 = and i32 %v1334, 31
  %v1336 = shl i32 %v1333, %v1335
  %v1337 = or i32 %v184, %v1336
  %v1338 = trunc i64 20 to i32
  %v1339 = and i32 %v1338, 31
  %v1340 = shl i32 %v189, %v1339
  %v1341 = or i32 %v1337, %v1340
  %v1342 = bitcast i32 %v1341 to float
  br label %bb205

bb205:                                            ; preds = %bb204, %bb203
  %v1343 = phi float [ %v1332, %bb203 ], [ %v1342, %bb204 ]
  %v1344 = fmul float %v1343, %v19
  %v1345 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 0
  br label %bb44

bb206:                                            ; preds = %bb44
  %v1346 = uitofp i32 %v206 to float
  %v1347 = fmul float %v1346, f0x3B000000
  %v1348 = icmp eq i32 %v201, 0
  br i1 %v1348, label %bb207, label %bb208

bb207:                                            ; preds = %bb206
  br label %bb209

bb208:                                            ; preds = %bb206
  %v1349 = fneg float %v1347
  br label %bb209

bb209:                                            ; preds = %bb208, %bb207
  %v1350 = phi float [ %v1347, %bb207 ], [ %v1349, %bb208 ]
  br label %bb211

bb210:                                            ; preds = %bb44
  %v1351 = add i32 %v205, 120
  %v1352 = trunc i64 23 to i32
  %v1353 = and i32 %v1352, 31
  %v1354 = shl i32 %v1351, %v1353
  %v1355 = or i32 %v201, %v1354
  %v1356 = trunc i64 20 to i32
  %v1357 = and i32 %v1356, 31
  %v1358 = shl i32 %v206, %v1357
  %v1359 = or i32 %v1355, %v1358
  %v1360 = bitcast i32 %v1359 to float
  br label %bb211

bb211:                                            ; preds = %bb210, %bb209
  %v1361 = phi float [ %v1350, %bb209 ], [ %v1360, %bb210 ]
  %v1362 = fmul float %v1361, %v19
  %v1363 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 1
  br label %bb45

bb212:                                            ; preds = %bb45
  %v1364 = uitofp i32 %v223 to float
  %v1365 = fmul float %v1364, f0x3B000000
  %v1366 = icmp eq i32 %v218, 0
  br i1 %v1366, label %bb213, label %bb214

bb213:                                            ; preds = %bb212
  br label %bb215

bb214:                                            ; preds = %bb212
  %v1367 = fneg float %v1365
  br label %bb215

bb215:                                            ; preds = %bb214, %bb213
  %v1368 = phi float [ %v1365, %bb213 ], [ %v1367, %bb214 ]
  br label %bb217

bb216:                                            ; preds = %bb45
  %v1369 = add i32 %v222, 120
  %v1370 = trunc i64 23 to i32
  %v1371 = and i32 %v1370, 31
  %v1372 = shl i32 %v1369, %v1371
  %v1373 = or i32 %v218, %v1372
  %v1374 = trunc i64 20 to i32
  %v1375 = and i32 %v1374, 31
  %v1376 = shl i32 %v223, %v1375
  %v1377 = or i32 %v1373, %v1376
  %v1378 = bitcast i32 %v1377 to float
  br label %bb217

bb217:                                            ; preds = %bb216, %bb215
  %v1379 = phi float [ %v1368, %bb215 ], [ %v1378, %bb216 ]
  %v1380 = fmul float %v1379, %v19
  %v1381 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 2
  br label %bb46

bb218:                                            ; preds = %bb46
  %v1382 = uitofp i32 %v240 to float
  %v1383 = fmul float %v1382, f0x3B000000
  %v1384 = icmp eq i32 %v235, 0
  br i1 %v1384, label %bb219, label %bb220

bb219:                                            ; preds = %bb218
  br label %bb221

bb220:                                            ; preds = %bb218
  %v1385 = fneg float %v1383
  br label %bb221

bb221:                                            ; preds = %bb220, %bb219
  %v1386 = phi float [ %v1383, %bb219 ], [ %v1385, %bb220 ]
  br label %bb223

bb222:                                            ; preds = %bb46
  %v1387 = add i32 %v239, 120
  %v1388 = trunc i64 23 to i32
  %v1389 = and i32 %v1388, 31
  %v1390 = shl i32 %v1387, %v1389
  %v1391 = or i32 %v235, %v1390
  %v1392 = trunc i64 20 to i32
  %v1393 = and i32 %v1392, 31
  %v1394 = shl i32 %v240, %v1393
  %v1395 = or i32 %v1391, %v1394
  %v1396 = bitcast i32 %v1395 to float
  br label %bb223

bb223:                                            ; preds = %bb222, %bb221
  %v1397 = phi float [ %v1386, %bb221 ], [ %v1396, %bb222 ]
  %v1398 = fmul float %v1397, %v19
  %v1399 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 3
  br label %bb47

bb224:                                            ; preds = %bb47
  %v1400 = uitofp i32 %v257 to float
  %v1401 = fmul float %v1400, f0x3B000000
  %v1402 = icmp eq i32 %v252, 0
  br i1 %v1402, label %bb225, label %bb226

bb225:                                            ; preds = %bb224
  br label %bb227

bb226:                                            ; preds = %bb224
  %v1403 = fneg float %v1401
  br label %bb227

bb227:                                            ; preds = %bb226, %bb225
  %v1404 = phi float [ %v1401, %bb225 ], [ %v1403, %bb226 ]
  br label %bb229

bb228:                                            ; preds = %bb47
  %v1405 = add i32 %v256, 120
  %v1406 = trunc i64 23 to i32
  %v1407 = and i32 %v1406, 31
  %v1408 = shl i32 %v1405, %v1407
  %v1409 = or i32 %v252, %v1408
  %v1410 = trunc i64 20 to i32
  %v1411 = and i32 %v1410, 31
  %v1412 = shl i32 %v257, %v1411
  %v1413 = or i32 %v1409, %v1412
  %v1414 = bitcast i32 %v1413 to float
  br label %bb229

bb229:                                            ; preds = %bb228, %bb227
  %v1415 = phi float [ %v1404, %bb227 ], [ %v1414, %bb228 ]
  %v1416 = fmul float %v1415, %v19
  %v1417 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 4
  br label %bb48

bb230:                                            ; preds = %bb48
  %v1418 = uitofp i32 %v274 to float
  %v1419 = fmul float %v1418, f0x3B000000
  %v1420 = icmp eq i32 %v269, 0
  br i1 %v1420, label %bb231, label %bb232

bb231:                                            ; preds = %bb230
  br label %bb233

bb232:                                            ; preds = %bb230
  %v1421 = fneg float %v1419
  br label %bb233

bb233:                                            ; preds = %bb232, %bb231
  %v1422 = phi float [ %v1419, %bb231 ], [ %v1421, %bb232 ]
  br label %bb235

bb234:                                            ; preds = %bb48
  %v1423 = add i32 %v273, 120
  %v1424 = trunc i64 23 to i32
  %v1425 = and i32 %v1424, 31
  %v1426 = shl i32 %v1423, %v1425
  %v1427 = or i32 %v269, %v1426
  %v1428 = trunc i64 20 to i32
  %v1429 = and i32 %v1428, 31
  %v1430 = shl i32 %v274, %v1429
  %v1431 = or i32 %v1427, %v1430
  %v1432 = bitcast i32 %v1431 to float
  br label %bb235

bb235:                                            ; preds = %bb234, %bb233
  %v1433 = phi float [ %v1422, %bb233 ], [ %v1432, %bb234 ]
  %v1434 = fmul float %v1433, %v19
  %v1435 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 5
  br label %bb49

bb236:                                            ; preds = %bb49
  %v1436 = uitofp i32 %v291 to float
  %v1437 = fmul float %v1436, f0x3B000000
  %v1438 = icmp eq i32 %v286, 0
  br i1 %v1438, label %bb237, label %bb238

bb237:                                            ; preds = %bb236
  br label %bb239

bb238:                                            ; preds = %bb236
  %v1439 = fneg float %v1437
  br label %bb239

bb239:                                            ; preds = %bb238, %bb237
  %v1440 = phi float [ %v1437, %bb237 ], [ %v1439, %bb238 ]
  br label %bb241

bb240:                                            ; preds = %bb49
  %v1441 = add i32 %v290, 120
  %v1442 = trunc i64 23 to i32
  %v1443 = and i32 %v1442, 31
  %v1444 = shl i32 %v1441, %v1443
  %v1445 = or i32 %v286, %v1444
  %v1446 = trunc i64 20 to i32
  %v1447 = and i32 %v1446, 31
  %v1448 = shl i32 %v291, %v1447
  %v1449 = or i32 %v1445, %v1448
  %v1450 = bitcast i32 %v1449 to float
  br label %bb241

bb241:                                            ; preds = %bb240, %bb239
  %v1451 = phi float [ %v1440, %bb239 ], [ %v1450, %bb240 ]
  %v1452 = fmul float %v1451, %v19
  %v1453 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 6
  br label %bb50

bb242:                                            ; preds = %bb50
  %v1454 = uitofp i32 %v308 to float
  %v1455 = fmul float %v1454, f0x3B000000
  %v1456 = icmp eq i32 %v303, 0
  br i1 %v1456, label %bb243, label %bb244

bb243:                                            ; preds = %bb242
  br label %bb245

bb244:                                            ; preds = %bb242
  %v1457 = fneg float %v1455
  br label %bb245

bb245:                                            ; preds = %bb244, %bb243
  %v1458 = phi float [ %v1455, %bb243 ], [ %v1457, %bb244 ]
  br label %bb247

bb246:                                            ; preds = %bb50
  %v1459 = add i32 %v307, 120
  %v1460 = trunc i64 23 to i32
  %v1461 = and i32 %v1460, 31
  %v1462 = shl i32 %v1459, %v1461
  %v1463 = or i32 %v303, %v1462
  %v1464 = trunc i64 20 to i32
  %v1465 = and i32 %v1464, 31
  %v1466 = shl i32 %v308, %v1465
  %v1467 = or i32 %v1463, %v1466
  %v1468 = bitcast i32 %v1467 to float
  br label %bb247

bb247:                                            ; preds = %bb246, %bb245
  %v1469 = phi float [ %v1458, %bb245 ], [ %v1468, %bb246 ]
  %v1470 = fmul float %v1469, %v19
  %v1471 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 7
  br label %bb51

bb248:                                            ; preds = %bb51
  %v1472 = uitofp i32 %v325 to float
  %v1473 = fmul float %v1472, f0x3B000000
  %v1474 = icmp eq i32 %v320, 0
  br i1 %v1474, label %bb249, label %bb250

bb249:                                            ; preds = %bb248
  br label %bb251

bb250:                                            ; preds = %bb248
  %v1475 = fneg float %v1473
  br label %bb251

bb251:                                            ; preds = %bb250, %bb249
  %v1476 = phi float [ %v1473, %bb249 ], [ %v1475, %bb250 ]
  br label %bb253

bb252:                                            ; preds = %bb51
  %v1477 = add i32 %v324, 120
  %v1478 = trunc i64 23 to i32
  %v1479 = and i32 %v1478, 31
  %v1480 = shl i32 %v1477, %v1479
  %v1481 = or i32 %v320, %v1480
  %v1482 = trunc i64 20 to i32
  %v1483 = and i32 %v1482, 31
  %v1484 = shl i32 %v325, %v1483
  %v1485 = or i32 %v1481, %v1484
  %v1486 = bitcast i32 %v1485 to float
  br label %bb253

bb253:                                            ; preds = %bb252, %bb251
  %v1487 = phi float [ %v1476, %bb251 ], [ %v1486, %bb252 ]
  %v1488 = fmul float %v1487, %v19
  %v1489 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 8
  br label %bb52

bb254:                                            ; preds = %bb52
  %v1490 = uitofp i32 %v342 to float
  %v1491 = fmul float %v1490, f0x3B000000
  %v1492 = icmp eq i32 %v337, 0
  br i1 %v1492, label %bb255, label %bb256

bb255:                                            ; preds = %bb254
  br label %bb257

bb256:                                            ; preds = %bb254
  %v1493 = fneg float %v1491
  br label %bb257

bb257:                                            ; preds = %bb256, %bb255
  %v1494 = phi float [ %v1491, %bb255 ], [ %v1493, %bb256 ]
  br label %bb259

bb258:                                            ; preds = %bb52
  %v1495 = add i32 %v341, 120
  %v1496 = trunc i64 23 to i32
  %v1497 = and i32 %v1496, 31
  %v1498 = shl i32 %v1495, %v1497
  %v1499 = or i32 %v337, %v1498
  %v1500 = trunc i64 20 to i32
  %v1501 = and i32 %v1500, 31
  %v1502 = shl i32 %v342, %v1501
  %v1503 = or i32 %v1499, %v1502
  %v1504 = bitcast i32 %v1503 to float
  br label %bb259

bb259:                                            ; preds = %bb258, %bb257
  %v1505 = phi float [ %v1494, %bb257 ], [ %v1504, %bb258 ]
  %v1506 = fmul float %v1505, %v19
  %v1507 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 9
  br label %bb53

bb260:                                            ; preds = %bb53
  %v1508 = uitofp i32 %v359 to float
  %v1509 = fmul float %v1508, f0x3B000000
  %v1510 = icmp eq i32 %v354, 0
  br i1 %v1510, label %bb261, label %bb262

bb261:                                            ; preds = %bb260
  br label %bb263

bb262:                                            ; preds = %bb260
  %v1511 = fneg float %v1509
  br label %bb263

bb263:                                            ; preds = %bb262, %bb261
  %v1512 = phi float [ %v1509, %bb261 ], [ %v1511, %bb262 ]
  br label %bb265

bb264:                                            ; preds = %bb53
  %v1513 = add i32 %v358, 120
  %v1514 = trunc i64 23 to i32
  %v1515 = and i32 %v1514, 31
  %v1516 = shl i32 %v1513, %v1515
  %v1517 = or i32 %v354, %v1516
  %v1518 = trunc i64 20 to i32
  %v1519 = and i32 %v1518, 31
  %v1520 = shl i32 %v359, %v1519
  %v1521 = or i32 %v1517, %v1520
  %v1522 = bitcast i32 %v1521 to float
  br label %bb265

bb265:                                            ; preds = %bb264, %bb263
  %v1523 = phi float [ %v1512, %bb263 ], [ %v1522, %bb264 ]
  %v1524 = fmul float %v1523, %v19
  %v1525 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 10
  br label %bb54

bb266:                                            ; preds = %bb54
  %v1526 = uitofp i32 %v376 to float
  %v1527 = fmul float %v1526, f0x3B000000
  %v1528 = icmp eq i32 %v371, 0
  br i1 %v1528, label %bb267, label %bb268

bb267:                                            ; preds = %bb266
  br label %bb269

bb268:                                            ; preds = %bb266
  %v1529 = fneg float %v1527
  br label %bb269

bb269:                                            ; preds = %bb268, %bb267
  %v1530 = phi float [ %v1527, %bb267 ], [ %v1529, %bb268 ]
  br label %bb271

bb270:                                            ; preds = %bb54
  %v1531 = add i32 %v375, 120
  %v1532 = trunc i64 23 to i32
  %v1533 = and i32 %v1532, 31
  %v1534 = shl i32 %v1531, %v1533
  %v1535 = or i32 %v371, %v1534
  %v1536 = trunc i64 20 to i32
  %v1537 = and i32 %v1536, 31
  %v1538 = shl i32 %v376, %v1537
  %v1539 = or i32 %v1535, %v1538
  %v1540 = bitcast i32 %v1539 to float
  br label %bb271

bb271:                                            ; preds = %bb270, %bb269
  %v1541 = phi float [ %v1530, %bb269 ], [ %v1540, %bb270 ]
  %v1542 = fmul float %v1541, %v19
  %v1543 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 11
  br label %bb55

bb272:                                            ; preds = %bb55
  %v1544 = uitofp i32 %v393 to float
  %v1545 = fmul float %v1544, f0x3B000000
  %v1546 = icmp eq i32 %v388, 0
  br i1 %v1546, label %bb273, label %bb274

bb273:                                            ; preds = %bb272
  br label %bb275

bb274:                                            ; preds = %bb272
  %v1547 = fneg float %v1545
  br label %bb275

bb275:                                            ; preds = %bb274, %bb273
  %v1548 = phi float [ %v1545, %bb273 ], [ %v1547, %bb274 ]
  br label %bb277

bb276:                                            ; preds = %bb55
  %v1549 = add i32 %v392, 120
  %v1550 = trunc i64 23 to i32
  %v1551 = and i32 %v1550, 31
  %v1552 = shl i32 %v1549, %v1551
  %v1553 = or i32 %v388, %v1552
  %v1554 = trunc i64 20 to i32
  %v1555 = and i32 %v1554, 31
  %v1556 = shl i32 %v393, %v1555
  %v1557 = or i32 %v1553, %v1556
  %v1558 = bitcast i32 %v1557 to float
  br label %bb277

bb277:                                            ; preds = %bb276, %bb275
  %v1559 = phi float [ %v1548, %bb275 ], [ %v1558, %bb276 ]
  %v1560 = fmul float %v1559, %v19
  %v1561 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 12
  br label %bb56

bb278:                                            ; preds = %bb56
  %v1562 = uitofp i32 %v410 to float
  %v1563 = fmul float %v1562, f0x3B000000
  %v1564 = icmp eq i32 %v405, 0
  br i1 %v1564, label %bb279, label %bb280

bb279:                                            ; preds = %bb278
  br label %bb281

bb280:                                            ; preds = %bb278
  %v1565 = fneg float %v1563
  br label %bb281

bb281:                                            ; preds = %bb280, %bb279
  %v1566 = phi float [ %v1563, %bb279 ], [ %v1565, %bb280 ]
  br label %bb283

bb282:                                            ; preds = %bb56
  %v1567 = add i32 %v409, 120
  %v1568 = trunc i64 23 to i32
  %v1569 = and i32 %v1568, 31
  %v1570 = shl i32 %v1567, %v1569
  %v1571 = or i32 %v405, %v1570
  %v1572 = trunc i64 20 to i32
  %v1573 = and i32 %v1572, 31
  %v1574 = shl i32 %v410, %v1573
  %v1575 = or i32 %v1571, %v1574
  %v1576 = bitcast i32 %v1575 to float
  br label %bb283

bb283:                                            ; preds = %bb282, %bb281
  %v1577 = phi float [ %v1566, %bb281 ], [ %v1576, %bb282 ]
  %v1578 = fmul float %v1577, %v19
  %v1579 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 13
  br label %bb57

bb284:                                            ; preds = %bb57
  %v1580 = uitofp i32 %v427 to float
  %v1581 = fmul float %v1580, f0x3B000000
  %v1582 = icmp eq i32 %v422, 0
  br i1 %v1582, label %bb285, label %bb286

bb285:                                            ; preds = %bb284
  br label %bb287

bb286:                                            ; preds = %bb284
  %v1583 = fneg float %v1581
  br label %bb287

bb287:                                            ; preds = %bb286, %bb285
  %v1584 = phi float [ %v1581, %bb285 ], [ %v1583, %bb286 ]
  br label %bb289

bb288:                                            ; preds = %bb57
  %v1585 = add i32 %v426, 120
  %v1586 = trunc i64 23 to i32
  %v1587 = and i32 %v1586, 31
  %v1588 = shl i32 %v1585, %v1587
  %v1589 = or i32 %v422, %v1588
  %v1590 = trunc i64 20 to i32
  %v1591 = and i32 %v1590, 31
  %v1592 = shl i32 %v427, %v1591
  %v1593 = or i32 %v1589, %v1592
  %v1594 = bitcast i32 %v1593 to float
  br label %bb289

bb289:                                            ; preds = %bb288, %bb287
  %v1595 = phi float [ %v1584, %bb287 ], [ %v1594, %bb288 ]
  %v1596 = fmul float %v1595, %v19
  %v1597 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 14
  br label %bb58

bb290:                                            ; preds = %bb58
  %v1598 = uitofp i32 %v444 to float
  %v1599 = fmul float %v1598, f0x3B000000
  %v1600 = icmp eq i32 %v439, 0
  br i1 %v1600, label %bb291, label %bb292

bb291:                                            ; preds = %bb290
  br label %bb293

bb292:                                            ; preds = %bb290
  %v1601 = fneg float %v1599
  br label %bb293

bb293:                                            ; preds = %bb292, %bb291
  %v1602 = phi float [ %v1599, %bb291 ], [ %v1601, %bb292 ]
  br label %bb295

bb294:                                            ; preds = %bb58
  %v1603 = add i32 %v443, 120
  %v1604 = trunc i64 23 to i32
  %v1605 = and i32 %v1604, 31
  %v1606 = shl i32 %v1603, %v1605
  %v1607 = or i32 %v439, %v1606
  %v1608 = trunc i64 20 to i32
  %v1609 = and i32 %v1608, 31
  %v1610 = shl i32 %v444, %v1609
  %v1611 = or i32 %v1607, %v1610
  %v1612 = bitcast i32 %v1611 to float
  br label %bb295

bb295:                                            ; preds = %bb294, %bb293
  %v1613 = phi float [ %v1602, %bb293 ], [ %v1612, %bb294 ]
  %v1614 = fmul float %v1613, %v19
  %v1615 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 15
  br label %bb59

bb296:                                            ; preds = %bb59
  %v1616 = uitofp i32 %v461 to float
  %v1617 = fmul float %v1616, f0x3B000000
  %v1618 = icmp eq i32 %v456, 0
  br i1 %v1618, label %bb297, label %bb298

bb297:                                            ; preds = %bb296
  br label %bb299

bb298:                                            ; preds = %bb296
  %v1619 = fneg float %v1617
  br label %bb299

bb299:                                            ; preds = %bb298, %bb297
  %v1620 = phi float [ %v1617, %bb297 ], [ %v1619, %bb298 ]
  br label %bb301

bb300:                                            ; preds = %bb59
  %v1621 = add i32 %v460, 120
  %v1622 = trunc i64 23 to i32
  %v1623 = and i32 %v1622, 31
  %v1624 = shl i32 %v1621, %v1623
  %v1625 = or i32 %v456, %v1624
  %v1626 = trunc i64 20 to i32
  %v1627 = and i32 %v1626, 31
  %v1628 = shl i32 %v461, %v1627
  %v1629 = or i32 %v1625, %v1628
  %v1630 = bitcast i32 %v1629 to float
  br label %bb301

bb301:                                            ; preds = %bb300, %bb299
  %v1631 = phi float [ %v1620, %bb299 ], [ %v1630, %bb300 ]
  %v1632 = fmul float %v1631, %v19
  %v1633 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 16
  br label %bb60

bb302:                                            ; preds = %bb60
  %v1634 = uitofp i32 %v478 to float
  %v1635 = fmul float %v1634, f0x3B000000
  %v1636 = icmp eq i32 %v473, 0
  br i1 %v1636, label %bb303, label %bb304

bb303:                                            ; preds = %bb302
  br label %bb305

bb304:                                            ; preds = %bb302
  %v1637 = fneg float %v1635
  br label %bb305

bb305:                                            ; preds = %bb304, %bb303
  %v1638 = phi float [ %v1635, %bb303 ], [ %v1637, %bb304 ]
  br label %bb307

bb306:                                            ; preds = %bb60
  %v1639 = add i32 %v477, 120
  %v1640 = trunc i64 23 to i32
  %v1641 = and i32 %v1640, 31
  %v1642 = shl i32 %v1639, %v1641
  %v1643 = or i32 %v473, %v1642
  %v1644 = trunc i64 20 to i32
  %v1645 = and i32 %v1644, 31
  %v1646 = shl i32 %v478, %v1645
  %v1647 = or i32 %v1643, %v1646
  %v1648 = bitcast i32 %v1647 to float
  br label %bb307

bb307:                                            ; preds = %bb306, %bb305
  %v1649 = phi float [ %v1638, %bb305 ], [ %v1648, %bb306 ]
  %v1650 = fmul float %v1649, %v19
  %v1651 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 17
  br label %bb61

bb308:                                            ; preds = %bb61
  %v1652 = uitofp i32 %v495 to float
  %v1653 = fmul float %v1652, f0x3B000000
  %v1654 = icmp eq i32 %v490, 0
  br i1 %v1654, label %bb309, label %bb310

bb309:                                            ; preds = %bb308
  br label %bb311

bb310:                                            ; preds = %bb308
  %v1655 = fneg float %v1653
  br label %bb311

bb311:                                            ; preds = %bb310, %bb309
  %v1656 = phi float [ %v1653, %bb309 ], [ %v1655, %bb310 ]
  br label %bb313

bb312:                                            ; preds = %bb61
  %v1657 = add i32 %v494, 120
  %v1658 = trunc i64 23 to i32
  %v1659 = and i32 %v1658, 31
  %v1660 = shl i32 %v1657, %v1659
  %v1661 = or i32 %v490, %v1660
  %v1662 = trunc i64 20 to i32
  %v1663 = and i32 %v1662, 31
  %v1664 = shl i32 %v495, %v1663
  %v1665 = or i32 %v1661, %v1664
  %v1666 = bitcast i32 %v1665 to float
  br label %bb313

bb313:                                            ; preds = %bb312, %bb311
  %v1667 = phi float [ %v1656, %bb311 ], [ %v1666, %bb312 ]
  %v1668 = fmul float %v1667, %v19
  %v1669 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 18
  br label %bb62

bb314:                                            ; preds = %bb62
  %v1670 = uitofp i32 %v512 to float
  %v1671 = fmul float %v1670, f0x3B000000
  %v1672 = icmp eq i32 %v507, 0
  br i1 %v1672, label %bb315, label %bb316

bb315:                                            ; preds = %bb314
  br label %bb317

bb316:                                            ; preds = %bb314
  %v1673 = fneg float %v1671
  br label %bb317

bb317:                                            ; preds = %bb316, %bb315
  %v1674 = phi float [ %v1671, %bb315 ], [ %v1673, %bb316 ]
  br label %bb319

bb318:                                            ; preds = %bb62
  %v1675 = add i32 %v511, 120
  %v1676 = trunc i64 23 to i32
  %v1677 = and i32 %v1676, 31
  %v1678 = shl i32 %v1675, %v1677
  %v1679 = or i32 %v507, %v1678
  %v1680 = trunc i64 20 to i32
  %v1681 = and i32 %v1680, 31
  %v1682 = shl i32 %v512, %v1681
  %v1683 = or i32 %v1679, %v1682
  %v1684 = bitcast i32 %v1683 to float
  br label %bb319

bb319:                                            ; preds = %bb318, %bb317
  %v1685 = phi float [ %v1674, %bb317 ], [ %v1684, %bb318 ]
  %v1686 = fmul float %v1685, %v19
  %v1687 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 19
  br label %bb63

bb320:                                            ; preds = %bb63
  %v1688 = uitofp i32 %v529 to float
  %v1689 = fmul float %v1688, f0x3B000000
  %v1690 = icmp eq i32 %v524, 0
  br i1 %v1690, label %bb321, label %bb322

bb321:                                            ; preds = %bb320
  br label %bb323

bb322:                                            ; preds = %bb320
  %v1691 = fneg float %v1689
  br label %bb323

bb323:                                            ; preds = %bb322, %bb321
  %v1692 = phi float [ %v1689, %bb321 ], [ %v1691, %bb322 ]
  br label %bb325

bb324:                                            ; preds = %bb63
  %v1693 = add i32 %v528, 120
  %v1694 = trunc i64 23 to i32
  %v1695 = and i32 %v1694, 31
  %v1696 = shl i32 %v1693, %v1695
  %v1697 = or i32 %v524, %v1696
  %v1698 = trunc i64 20 to i32
  %v1699 = and i32 %v1698, 31
  %v1700 = shl i32 %v529, %v1699
  %v1701 = or i32 %v1697, %v1700
  %v1702 = bitcast i32 %v1701 to float
  br label %bb325

bb325:                                            ; preds = %bb324, %bb323
  %v1703 = phi float [ %v1692, %bb323 ], [ %v1702, %bb324 ]
  %v1704 = fmul float %v1703, %v19
  %v1705 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 20
  br label %bb64

bb326:                                            ; preds = %bb64
  %v1706 = uitofp i32 %v546 to float
  %v1707 = fmul float %v1706, f0x3B000000
  %v1708 = icmp eq i32 %v541, 0
  br i1 %v1708, label %bb327, label %bb328

bb327:                                            ; preds = %bb326
  br label %bb329

bb328:                                            ; preds = %bb326
  %v1709 = fneg float %v1707
  br label %bb329

bb329:                                            ; preds = %bb328, %bb327
  %v1710 = phi float [ %v1707, %bb327 ], [ %v1709, %bb328 ]
  br label %bb331

bb330:                                            ; preds = %bb64
  %v1711 = add i32 %v545, 120
  %v1712 = trunc i64 23 to i32
  %v1713 = and i32 %v1712, 31
  %v1714 = shl i32 %v1711, %v1713
  %v1715 = or i32 %v541, %v1714
  %v1716 = trunc i64 20 to i32
  %v1717 = and i32 %v1716, 31
  %v1718 = shl i32 %v546, %v1717
  %v1719 = or i32 %v1715, %v1718
  %v1720 = bitcast i32 %v1719 to float
  br label %bb331

bb331:                                            ; preds = %bb330, %bb329
  %v1721 = phi float [ %v1710, %bb329 ], [ %v1720, %bb330 ]
  %v1722 = fmul float %v1721, %v19
  %v1723 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 21
  br label %bb65

bb332:                                            ; preds = %bb65
  %v1724 = uitofp i32 %v563 to float
  %v1725 = fmul float %v1724, f0x3B000000
  %v1726 = icmp eq i32 %v558, 0
  br i1 %v1726, label %bb333, label %bb334

bb333:                                            ; preds = %bb332
  br label %bb335

bb334:                                            ; preds = %bb332
  %v1727 = fneg float %v1725
  br label %bb335

bb335:                                            ; preds = %bb334, %bb333
  %v1728 = phi float [ %v1725, %bb333 ], [ %v1727, %bb334 ]
  br label %bb337

bb336:                                            ; preds = %bb65
  %v1729 = add i32 %v562, 120
  %v1730 = trunc i64 23 to i32
  %v1731 = and i32 %v1730, 31
  %v1732 = shl i32 %v1729, %v1731
  %v1733 = or i32 %v558, %v1732
  %v1734 = trunc i64 20 to i32
  %v1735 = and i32 %v1734, 31
  %v1736 = shl i32 %v563, %v1735
  %v1737 = or i32 %v1733, %v1736
  %v1738 = bitcast i32 %v1737 to float
  br label %bb337

bb337:                                            ; preds = %bb336, %bb335
  %v1739 = phi float [ %v1728, %bb335 ], [ %v1738, %bb336 ]
  %v1740 = fmul float %v1739, %v19
  %v1741 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 22
  br label %bb66

bb338:                                            ; preds = %bb66
  %v1742 = uitofp i32 %v580 to float
  %v1743 = fmul float %v1742, f0x3B000000
  %v1744 = icmp eq i32 %v575, 0
  br i1 %v1744, label %bb339, label %bb340

bb339:                                            ; preds = %bb338
  br label %bb341

bb340:                                            ; preds = %bb338
  %v1745 = fneg float %v1743
  br label %bb341

bb341:                                            ; preds = %bb340, %bb339
  %v1746 = phi float [ %v1743, %bb339 ], [ %v1745, %bb340 ]
  br label %bb343

bb342:                                            ; preds = %bb66
  %v1747 = add i32 %v579, 120
  %v1748 = trunc i64 23 to i32
  %v1749 = and i32 %v1748, 31
  %v1750 = shl i32 %v1747, %v1749
  %v1751 = or i32 %v575, %v1750
  %v1752 = trunc i64 20 to i32
  %v1753 = and i32 %v1752, 31
  %v1754 = shl i32 %v580, %v1753
  %v1755 = or i32 %v1751, %v1754
  %v1756 = bitcast i32 %v1755 to float
  br label %bb343

bb343:                                            ; preds = %bb342, %bb341
  %v1757 = phi float [ %v1746, %bb341 ], [ %v1756, %bb342 ]
  %v1758 = fmul float %v1757, %v19
  %v1759 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 23
  br label %bb67

bb344:                                            ; preds = %bb67
  %v1760 = uitofp i32 %v597 to float
  %v1761 = fmul float %v1760, f0x3B000000
  %v1762 = icmp eq i32 %v592, 0
  br i1 %v1762, label %bb345, label %bb346

bb345:                                            ; preds = %bb344
  br label %bb347

bb346:                                            ; preds = %bb344
  %v1763 = fneg float %v1761
  br label %bb347

bb347:                                            ; preds = %bb346, %bb345
  %v1764 = phi float [ %v1761, %bb345 ], [ %v1763, %bb346 ]
  br label %bb349

bb348:                                            ; preds = %bb67
  %v1765 = add i32 %v596, 120
  %v1766 = trunc i64 23 to i32
  %v1767 = and i32 %v1766, 31
  %v1768 = shl i32 %v1765, %v1767
  %v1769 = or i32 %v592, %v1768
  %v1770 = trunc i64 20 to i32
  %v1771 = and i32 %v1770, 31
  %v1772 = shl i32 %v597, %v1771
  %v1773 = or i32 %v1769, %v1772
  %v1774 = bitcast i32 %v1773 to float
  br label %bb349

bb349:                                            ; preds = %bb348, %bb347
  %v1775 = phi float [ %v1764, %bb347 ], [ %v1774, %bb348 ]
  %v1776 = fmul float %v1775, %v19
  %v1777 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 24
  br label %bb68

bb350:                                            ; preds = %bb68
  %v1778 = uitofp i32 %v614 to float
  %v1779 = fmul float %v1778, f0x3B000000
  %v1780 = icmp eq i32 %v609, 0
  br i1 %v1780, label %bb351, label %bb352

bb351:                                            ; preds = %bb350
  br label %bb353

bb352:                                            ; preds = %bb350
  %v1781 = fneg float %v1779
  br label %bb353

bb353:                                            ; preds = %bb352, %bb351
  %v1782 = phi float [ %v1779, %bb351 ], [ %v1781, %bb352 ]
  br label %bb355

bb354:                                            ; preds = %bb68
  %v1783 = add i32 %v613, 120
  %v1784 = trunc i64 23 to i32
  %v1785 = and i32 %v1784, 31
  %v1786 = shl i32 %v1783, %v1785
  %v1787 = or i32 %v609, %v1786
  %v1788 = trunc i64 20 to i32
  %v1789 = and i32 %v1788, 31
  %v1790 = shl i32 %v614, %v1789
  %v1791 = or i32 %v1787, %v1790
  %v1792 = bitcast i32 %v1791 to float
  br label %bb355

bb355:                                            ; preds = %bb354, %bb353
  %v1793 = phi float [ %v1782, %bb353 ], [ %v1792, %bb354 ]
  %v1794 = fmul float %v1793, %v19
  %v1795 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 25
  br label %bb69

bb356:                                            ; preds = %bb69
  %v1796 = uitofp i32 %v631 to float
  %v1797 = fmul float %v1796, f0x3B000000
  %v1798 = icmp eq i32 %v626, 0
  br i1 %v1798, label %bb357, label %bb358

bb357:                                            ; preds = %bb356
  br label %bb359

bb358:                                            ; preds = %bb356
  %v1799 = fneg float %v1797
  br label %bb359

bb359:                                            ; preds = %bb358, %bb357
  %v1800 = phi float [ %v1797, %bb357 ], [ %v1799, %bb358 ]
  br label %bb361

bb360:                                            ; preds = %bb69
  %v1801 = add i32 %v630, 120
  %v1802 = trunc i64 23 to i32
  %v1803 = and i32 %v1802, 31
  %v1804 = shl i32 %v1801, %v1803
  %v1805 = or i32 %v626, %v1804
  %v1806 = trunc i64 20 to i32
  %v1807 = and i32 %v1806, 31
  %v1808 = shl i32 %v631, %v1807
  %v1809 = or i32 %v1805, %v1808
  %v1810 = bitcast i32 %v1809 to float
  br label %bb361

bb361:                                            ; preds = %bb360, %bb359
  %v1811 = phi float [ %v1800, %bb359 ], [ %v1810, %bb360 ]
  %v1812 = fmul float %v1811, %v19
  %v1813 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 26
  br label %bb70

bb362:                                            ; preds = %bb70
  %v1814 = uitofp i32 %v648 to float
  %v1815 = fmul float %v1814, f0x3B000000
  %v1816 = icmp eq i32 %v643, 0
  br i1 %v1816, label %bb363, label %bb364

bb363:                                            ; preds = %bb362
  br label %bb365

bb364:                                            ; preds = %bb362
  %v1817 = fneg float %v1815
  br label %bb365

bb365:                                            ; preds = %bb364, %bb363
  %v1818 = phi float [ %v1815, %bb363 ], [ %v1817, %bb364 ]
  br label %bb367

bb366:                                            ; preds = %bb70
  %v1819 = add i32 %v647, 120
  %v1820 = trunc i64 23 to i32
  %v1821 = and i32 %v1820, 31
  %v1822 = shl i32 %v1819, %v1821
  %v1823 = or i32 %v643, %v1822
  %v1824 = trunc i64 20 to i32
  %v1825 = and i32 %v1824, 31
  %v1826 = shl i32 %v648, %v1825
  %v1827 = or i32 %v1823, %v1826
  %v1828 = bitcast i32 %v1827 to float
  br label %bb367

bb367:                                            ; preds = %bb366, %bb365
  %v1829 = phi float [ %v1818, %bb365 ], [ %v1828, %bb366 ]
  %v1830 = fmul float %v1829, %v19
  %v1831 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 27
  br label %bb71

bb368:                                            ; preds = %bb71
  %v1832 = uitofp i32 %v665 to float
  %v1833 = fmul float %v1832, f0x3B000000
  %v1834 = icmp eq i32 %v660, 0
  br i1 %v1834, label %bb369, label %bb370

bb369:                                            ; preds = %bb368
  br label %bb371

bb370:                                            ; preds = %bb368
  %v1835 = fneg float %v1833
  br label %bb371

bb371:                                            ; preds = %bb370, %bb369
  %v1836 = phi float [ %v1833, %bb369 ], [ %v1835, %bb370 ]
  br label %bb373

bb372:                                            ; preds = %bb71
  %v1837 = add i32 %v664, 120
  %v1838 = trunc i64 23 to i32
  %v1839 = and i32 %v1838, 31
  %v1840 = shl i32 %v1837, %v1839
  %v1841 = or i32 %v660, %v1840
  %v1842 = trunc i64 20 to i32
  %v1843 = and i32 %v1842, 31
  %v1844 = shl i32 %v665, %v1843
  %v1845 = or i32 %v1841, %v1844
  %v1846 = bitcast i32 %v1845 to float
  br label %bb373

bb373:                                            ; preds = %bb372, %bb371
  %v1847 = phi float [ %v1836, %bb371 ], [ %v1846, %bb372 ]
  %v1848 = fmul float %v1847, %v19
  %v1849 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 28
  br label %bb72

bb374:                                            ; preds = %bb72
  %v1850 = uitofp i32 %v682 to float
  %v1851 = fmul float %v1850, f0x3B000000
  %v1852 = icmp eq i32 %v677, 0
  br i1 %v1852, label %bb375, label %bb376

bb375:                                            ; preds = %bb374
  br label %bb377

bb376:                                            ; preds = %bb374
  %v1853 = fneg float %v1851
  br label %bb377

bb377:                                            ; preds = %bb376, %bb375
  %v1854 = phi float [ %v1851, %bb375 ], [ %v1853, %bb376 ]
  br label %bb379

bb378:                                            ; preds = %bb72
  %v1855 = add i32 %v681, 120
  %v1856 = trunc i64 23 to i32
  %v1857 = and i32 %v1856, 31
  %v1858 = shl i32 %v1855, %v1857
  %v1859 = or i32 %v677, %v1858
  %v1860 = trunc i64 20 to i32
  %v1861 = and i32 %v1860, 31
  %v1862 = shl i32 %v682, %v1861
  %v1863 = or i32 %v1859, %v1862
  %v1864 = bitcast i32 %v1863 to float
  br label %bb379

bb379:                                            ; preds = %bb378, %bb377
  %v1865 = phi float [ %v1854, %bb377 ], [ %v1864, %bb378 ]
  %v1866 = fmul float %v1865, %v19
  %v1867 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 29
  br label %bb73

bb380:                                            ; preds = %bb73
  %v1868 = uitofp i32 %v699 to float
  %v1869 = fmul float %v1868, f0x3B000000
  %v1870 = icmp eq i32 %v694, 0
  br i1 %v1870, label %bb381, label %bb382

bb381:                                            ; preds = %bb380
  br label %bb383

bb382:                                            ; preds = %bb380
  %v1871 = fneg float %v1869
  br label %bb383

bb383:                                            ; preds = %bb382, %bb381
  %v1872 = phi float [ %v1869, %bb381 ], [ %v1871, %bb382 ]
  br label %bb385

bb384:                                            ; preds = %bb73
  %v1873 = add i32 %v698, 120
  %v1874 = trunc i64 23 to i32
  %v1875 = and i32 %v1874, 31
  %v1876 = shl i32 %v1873, %v1875
  %v1877 = or i32 %v694, %v1876
  %v1878 = trunc i64 20 to i32
  %v1879 = and i32 %v1878, 31
  %v1880 = shl i32 %v699, %v1879
  %v1881 = or i32 %v1877, %v1880
  %v1882 = bitcast i32 %v1881 to float
  br label %bb385

bb385:                                            ; preds = %bb384, %bb383
  %v1883 = phi float [ %v1872, %bb383 ], [ %v1882, %bb384 ]
  %v1884 = fmul float %v1883, %v19
  %v1885 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 30
  br label %bb74

bb386:                                            ; preds = %bb74
  %v1886 = uitofp i32 %v716 to float
  %v1887 = fmul float %v1886, f0x3B000000
  %v1888 = icmp eq i32 %v711, 0
  br i1 %v1888, label %bb387, label %bb388

bb387:                                            ; preds = %bb386
  br label %bb389

bb388:                                            ; preds = %bb386
  %v1889 = fneg float %v1887
  br label %bb389

bb389:                                            ; preds = %bb388, %bb387
  %v1890 = phi float [ %v1887, %bb387 ], [ %v1889, %bb388 ]
  br label %bb391

bb390:                                            ; preds = %bb74
  %v1891 = add i32 %v715, 120
  %v1892 = trunc i64 23 to i32
  %v1893 = and i32 %v1892, 31
  %v1894 = shl i32 %v1891, %v1893
  %v1895 = or i32 %v711, %v1894
  %v1896 = trunc i64 20 to i32
  %v1897 = and i32 %v1896, 31
  %v1898 = shl i32 %v716, %v1897
  %v1899 = or i32 %v1895, %v1898
  %v1900 = bitcast i32 %v1899 to float
  br label %bb391

bb391:                                            ; preds = %bb390, %bb389
  %v1901 = phi float [ %v1890, %bb389 ], [ %v1900, %bb390 ]
  %v1902 = fmul float %v1901, %v19
  %v1903 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23b72fdefd237786_2, i64 31
  br label %bb75
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare float @llvm.nvvm.shfl.sync.bfly.f32(i32, float, i32, i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare float @llvm.nvvm.shfl.sync.idx.f32(i32, float, i32, i32) #3

; Function Attrs: alwaysinline nounwind
define internal float @__nv_expf(float %a) #4 {
  %1 = bitcast i32 1069066811 to float
  %2 = bitcast i32 849703008 to float
  %3 = bitcast i32 1262485504 to float
  %4 = fdiv float %1, 2.520000e+02
  %5 = call i32 @__nvvm_reflect(ptr @.str) #8
  %6 = icmp ne i32 %5, 0
  br i1 %6, label %7, label %9

7:                                                ; preds = %0
  %8 = call float @llvm.nvvm.fma.rn.ftz.f(float %a, float %4, float 5.000000e-01) #8
  br label %__nv_fmaf_rn.exit.i.i

9:                                                ; preds = %0
  %10 = call float @llvm.nvvm.fma.rn.f(float %a, float %4, float 5.000000e-01) #8
  br label %__nv_fmaf_rn.exit.i.i

__nv_fmaf_rn.exit.i.i:                            ; preds = %9, %7
  %.02 = phi float [ %8, %7 ], [ %10, %9 ]
  %11 = call i32 @__nvvm_reflect(ptr @.str) #8
  %12 = icmp ne i32 %11, 0
  br i1 %12, label %13, label %15

13:                                               ; preds = %__nv_fmaf_rn.exit.i.i
  %14 = call float @llvm.nvvm.saturate.ftz.f(float %.02) #8
  br label %__nv_saturatef.exit.i.i

15:                                               ; preds = %__nv_fmaf_rn.exit.i.i
  %16 = call float @llvm.nvvm.saturate.f(float %.02) #8
  br label %__nv_saturatef.exit.i.i

__nv_saturatef.exit.i.i:                          ; preds = %15, %13
  %.03 = phi float [ %14, %13 ], [ %16, %15 ]
  %17 = fadd float -1.260000e+02, %3
  %18 = fadd float %17, 1.270000e+02
  %19 = call i32 @__nvvm_reflect(ptr @.str) #8
  %20 = icmp ne i32 %19, 0
  br i1 %20, label %21, label %23

21:                                               ; preds = %__nv_saturatef.exit.i.i
  %22 = call float @llvm.nvvm.fma.rm.ftz.f(float %.03, float 2.520000e+02, float %18) #8
  br label %__nv_fmaf_rd.exit.i.i

23:                                               ; preds = %__nv_saturatef.exit.i.i
  %24 = call float @llvm.nvvm.fma.rm.f(float %.03, float 2.520000e+02, float %18) #8
  br label %__nv_fmaf_rd.exit.i.i

__nv_fmaf_rd.exit.i.i:                            ; preds = %23, %21
  %.04 = phi float [ %22, %21 ], [ %24, %23 ]
  %25 = fadd float %3, 1.270000e+02
  %26 = fsub float %.04, %25
  %27 = fsub float -0.000000e+00, %26
  %28 = call i32 @__nvvm_reflect(ptr @.str) #8
  %29 = icmp ne i32 %28, 0
  br i1 %29, label %30, label %32

30:                                               ; preds = %__nv_fmaf_rd.exit.i.i
  %31 = call float @llvm.nvvm.fma.rn.ftz.f(float %a, float %1, float %27) #8
  br label %__internal_fmad.exit3.i.i

32:                                               ; preds = %__nv_fmaf_rd.exit.i.i
  %33 = call float @llvm.nvvm.fma.rn.f(float %a, float %1, float %27) #8
  br label %__internal_fmad.exit3.i.i

__internal_fmad.exit3.i.i:                        ; preds = %32, %30
  %.0 = phi float [ %31, %30 ], [ %33, %32 ]
  %34 = call i32 @__nvvm_reflect(ptr @.str) #8
  %35 = icmp ne i32 %34, 0
  br i1 %35, label %36, label %38

36:                                               ; preds = %__internal_fmad.exit3.i.i
  %37 = call float @llvm.nvvm.fma.rn.ftz.f(float %a, float %2, float %.0) #8
  br label %__internal_accurate_expf.exit

38:                                               ; preds = %__internal_fmad.exit3.i.i
  %39 = call float @llvm.nvvm.fma.rn.f(float %a, float %2, float %.0) #8
  br label %__internal_accurate_expf.exit

__internal_accurate_expf.exit:                    ; preds = %36, %38
  %.01 = phi float [ %37, %36 ], [ %39, %38 ]
  %40 = bitcast float %.04 to i32
  %41 = shl i32 %40, 23
  %42 = bitcast i32 %41 to float
  %43 = call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01)
  %44 = fmul float %43, %42
  ret float %44
}

declare i32 @__nvvm_reflect(ptr) #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fma.rn.ftz.f(float, float, float) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fma.rn.f(float, float, float) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.saturate.ftz.f(float) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.saturate.f(float) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fma.rm.ftz.f(float, float, float) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fma.rm.f(float, float, float) #6

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.ex2.approx.ftz.f32(float) #7

attributes #0 = { convergent }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }
attributes #3 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #4 = { alwaysinline nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #8 = { nounwind }

!llvm.ident = !{!0}
!nvvmir.version = !{!1}

!0 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!1 = !{i32 2, i32 0}
