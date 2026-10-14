; ModuleID = 'builtin.module'
source_filename = "oxide_gdn_simt"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; shared source: oxide_gdn_simt::kernels::cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_gdn_f32_128::READ
@__shared_mem_a858d0f5fb2dcc30_2 = addrspace(3) global [128 x float] zeroinitializer, align 4
; shared source: oxide_gdn_simt::kernels::cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_gdn_f32_128::RESULT
@__shared_mem_a858d0f5fb2dcc30_1 = addrspace(3) global [8 x float] zeroinitializer, align 4
; shared source: oxide_gdn_simt::kernels::cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_gdn_f32_128::PRED
@__shared_mem_a858d0f5fb2dcc30_0 = addrspace(3) global [256 x float] zeroinitializer, align 4
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #0
declare float @llvm.nvvm.shfl.sync.bfly.f32(i32, float, i32, i32) #0

define ptx_kernel void @gdn_f32_128(ptr %v0, ptr %v1, ptr %v2, ptr %v3, ptr %v4, ptr %v5, ptr %v6, i32 %v7) #0 {
entry:
  br label %bb0
bb0:
  %v8 = phi ptr [ %v0, %entry ]
  %v9 = phi ptr [ %v1, %entry ]
  %v10 = phi ptr [ %v2, %entry ]
  %v11 = phi ptr [ %v3, %entry ]
  %v12 = phi ptr [ %v4, %entry ]
  %v13 = phi ptr [ %v5, %entry ]
  %v14 = phi ptr [ %v6, %entry ]
  %v15 = phi i32 [ %v7, %entry ]
  %v16 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v17 = zext i32 %v16 to i64
  %v18 = udiv i64 %v17, 8
  %v19 = urem i64 %v17, 8
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v21 = zext i32 %v20 to i64
  %v22 = udiv i64 %v21, 16
  %v23 = urem i64 %v21, 16
  %v24 = mul i64 %v23, 8
  %v25 = add i64 %v24, %v19
  %v26 = udiv i64 %v22, 3
  %v27 = zext i32 %v15 to i64
  %v28 = getelementptr inbounds i32, ptr %v14, i64 1
  %v29 = load i32, ptr %v28, align 4
  %v30 = load i32, ptr %v14, align 4
  %v31 = getelementptr inbounds i32, ptr %v14, i64 2
  %v32 = load i32, ptr %v31, align 4
  %v33 = add i32 %v30, %v32
  %v34 = icmp slt i32 %v33, 0
  %v35 = xor i1 %v34, 1
  br i1 %v35, label %bb4, label %bb3
bb3:
  %v36 = sub i32 0, %v33
  %v37 = sext i32 %v36 to i64
  br label %bb5
bb4:
  br label %bb5
bb5:
  %v38 = phi i64 [ %v37, %bb3 ], [ 0, %bb4 ]
  %v39 = icmp sle i32 %v29, 0
  %v40 = xor i1 %v39, 1
  br i1 %v40, label %bb7, label %bb6
bb6:
  br label %bb11
bb7:
  %v41 = sext i32 %v29 to i64
  %v42 = icmp ugt i64 %v41, %v27
  %v43 = xor i1 %v42, 1
  br i1 %v43, label %bb9, label %bb8
bb8:
  br label %bb10
bb9:
  br label %bb10
bb10:
  %v44 = phi i64 [ %v27, %bb8 ], [ %v41, %bb9 ]
  br label %bb11
bb11:
  %v45 = phi i64 [ 0, %bb6 ], [ %v44, %bb10 ]
  br label %bb12
bb12:
  %v46 = phi i64 [ %v18, %bb11 ], [ %v54, %bb13 ]
  %v47 = icmp ult i64 %v46, %v27
  %v48 = xor i1 %v47, 1
  br i1 %v48, label %bb14, label %bb13
bb13:
  %v49 = mul i64 %v22, %v27
  %v50 = add i64 %v49, %v46
  %v51 = mul i64 %v50, 128
  %v52 = add i64 %v51, %v25
  %v53 = getelementptr inbounds float, ptr %v9, i64 %v52
  store float 0.0, ptr %v53, align 4
  %v54 = add i64 %v46, 16
  br label %bb12
bb14:
  %v55 = mul i64 %v22, 128
  %v56 = add i64 %v55, %v18
  %v57 = mul i64 %v56, 128
  %v58 = add i64 %v57, %v25
  %v59 = getelementptr inbounds float, ptr %v8, i64 %v58
  %v60 = load float, ptr %v59, align 4
  %v61 = add i64 %v56, 16
  %v62 = mul i64 %v61, 128
  %v63 = add i64 %v62, %v25
  %v64 = getelementptr inbounds float, ptr %v8, i64 %v63
  %v65 = load float, ptr %v64, align 4
  %v66 = add i64 %v56, 32
  %v67 = mul i64 %v66, 128
  %v68 = add i64 %v67, %v25
  %v69 = getelementptr inbounds float, ptr %v8, i64 %v68
  %v70 = load float, ptr %v69, align 4
  %v71 = add i64 %v56, 48
  %v72 = mul i64 %v71, 128
  %v73 = add i64 %v72, %v25
  %v74 = getelementptr inbounds float, ptr %v8, i64 %v73
  %v75 = load float, ptr %v74, align 4
  %v76 = add i64 %v56, 64
  %v77 = mul i64 %v76, 128
  %v78 = add i64 %v77, %v25
  %v79 = getelementptr inbounds float, ptr %v8, i64 %v78
  %v80 = load float, ptr %v79, align 4
  %v81 = add i64 %v56, 80
  %v82 = mul i64 %v81, 128
  %v83 = add i64 %v82, %v25
  %v84 = getelementptr inbounds float, ptr %v8, i64 %v83
  %v85 = load float, ptr %v84, align 4
  %v86 = add i64 %v56, 96
  %v87 = mul i64 %v86, 128
  %v88 = add i64 %v87, %v25
  %v89 = getelementptr inbounds float, ptr %v8, i64 %v88
  %v90 = load float, ptr %v89, align 4
  %v91 = add i64 %v56, 112
  %v92 = mul i64 %v91, 128
  %v93 = add i64 %v92, %v25
  %v94 = getelementptr inbounds float, ptr %v8, i64 %v93
  %v95 = load float, ptr %v94, align 4
  br label %bb15
bb15:
  %v96 = phi i64 [ %v38, %bb14 ], [ %v386, %bb44 ]
  %v97 = phi float [ %v60, %bb14 ], [ %v326, %bb44 ]
  %v98 = phi float [ %v65, %bb14 ], [ %v329, %bb44 ]
  %v99 = phi float [ %v70, %bb14 ], [ %v332, %bb44 ]
  %v100 = phi float [ %v75, %bb14 ], [ %v335, %bb44 ]
  %v101 = phi float [ %v80, %bb14 ], [ %v338, %bb44 ]
  %v102 = phi float [ %v85, %bb14 ], [ %v341, %bb44 ]
  %v103 = phi float [ %v90, %bb14 ], [ %v344, %bb44 ]
  %v104 = phi float [ %v95, %bb14 ], [ %v347, %bb44 ]
  %v105 = icmp ult i64 %v96, %v45
  %v106 = xor i1 %v105, 1
  br i1 %v106, label %bb45, label %bb16
bb16:
  %v107 = mul i64 %v96, 48
  %v108 = add i64 %v107, %v22
  %v109 = getelementptr inbounds float, ptr %v13, i64 %v108
  %v110 = load float, ptr %v109, align 4
  %v111 = mul i64 %v96, 48
  %v112 = add i64 %v111, %v22
  %v113 = getelementptr inbounds float, ptr %v12, i64 %v112
  %v114 = load float, ptr %v113, align 4
  %v115 = mul i64 %v96, 80
  %v116 = add i64 %v115, 32
  %v117 = add i64 %v116, %v22
  %v118 = mul i64 %v117, 128
  %v119 = add i64 %v118, %v25
  %v120 = getelementptr inbounds float, ptr %v10, i64 %v119
  %v121 = load float, ptr %v120, align 4
  %v122 = mul i64 %v96, 32
  %v123 = add i64 %v122, %v26
  %v124 = mul i64 %v123, 128
  %v125 = add i64 %v124, %v18
  %v126 = getelementptr inbounds float, ptr %v11, i64 %v125
  %v127 = load float, ptr %v126, align 4
  %v128 = mul i64 %v96, 32
  %v129 = add i64 %v128, 16
  %v130 = add i64 %v129, %v26
  %v131 = mul i64 %v130, 128
  %v132 = add i64 %v131, %v18
  %v133 = getelementptr inbounds float, ptr %v11, i64 %v132
  %v134 = load float, ptr %v133, align 4
  %v135 = fmul float %v97, %v110
  %v136 = fmul float %v135, %v134
  %v137 = mul i64 %v96, 32
  %v138 = add i64 %v137, %v26
  %v139 = mul i64 %v138, 128
  %v140 = add i64 %v139, %v18
  %v141 = add i64 %v140, 16
  %v142 = getelementptr inbounds float, ptr %v11, i64 %v141
  %v143 = load float, ptr %v142, align 4
  %v144 = mul i64 %v96, 32
  %v145 = add i64 %v144, 16
  %v146 = add i64 %v145, %v26
  %v147 = mul i64 %v146, 128
  %v148 = add i64 %v147, %v18
  %v149 = add i64 %v148, 16
  %v150 = getelementptr inbounds float, ptr %v11, i64 %v149
  %v151 = load float, ptr %v150, align 4
  %v152 = fmul float %v98, %v110
  %v153 = fmul float %v152, %v151
  %v154 = mul i64 %v96, 32
  %v155 = add i64 %v154, %v26
  %v156 = mul i64 %v155, 128
  %v157 = add i64 %v156, %v18
  %v158 = add i64 %v157, 32
  %v159 = getelementptr inbounds float, ptr %v11, i64 %v158
  %v160 = load float, ptr %v159, align 4
  %v161 = mul i64 %v96, 32
  %v162 = add i64 %v161, 16
  %v163 = add i64 %v162, %v26
  %v164 = mul i64 %v163, 128
  %v165 = add i64 %v164, %v18
  %v166 = add i64 %v165, 32
  %v167 = getelementptr inbounds float, ptr %v11, i64 %v166
  %v168 = load float, ptr %v167, align 4
  %v169 = fmul float %v99, %v110
  %v170 = fmul float %v169, %v168
  %v171 = mul i64 %v96, 32
  %v172 = add i64 %v171, %v26
  %v173 = mul i64 %v172, 128
  %v174 = add i64 %v173, %v18
  %v175 = add i64 %v174, 48
  %v176 = getelementptr inbounds float, ptr %v11, i64 %v175
  %v177 = load float, ptr %v176, align 4
  %v178 = mul i64 %v96, 32
  %v179 = add i64 %v178, 16
  %v180 = add i64 %v179, %v26
  %v181 = mul i64 %v180, 128
  %v182 = add i64 %v181, %v18
  %v183 = add i64 %v182, 48
  %v184 = getelementptr inbounds float, ptr %v11, i64 %v183
  %v185 = load float, ptr %v184, align 4
  %v186 = fmul float %v100, %v110
  %v187 = fmul float %v186, %v185
  %v188 = mul i64 %v96, 32
  %v189 = add i64 %v188, %v26
  %v190 = mul i64 %v189, 128
  %v191 = add i64 %v190, %v18
  %v192 = add i64 %v191, 64
  %v193 = getelementptr inbounds float, ptr %v11, i64 %v192
  %v194 = load float, ptr %v193, align 4
  %v195 = mul i64 %v96, 32
  %v196 = add i64 %v195, 16
  %v197 = add i64 %v196, %v26
  %v198 = mul i64 %v197, 128
  %v199 = add i64 %v198, %v18
  %v200 = add i64 %v199, 64
  %v201 = getelementptr inbounds float, ptr %v11, i64 %v200
  %v202 = load float, ptr %v201, align 4
  %v203 = fmul float %v101, %v110
  %v204 = fmul float %v203, %v202
  %v205 = mul i64 %v96, 32
  %v206 = add i64 %v205, %v26
  %v207 = mul i64 %v206, 128
  %v208 = add i64 %v207, %v18
  %v209 = add i64 %v208, 80
  %v210 = getelementptr inbounds float, ptr %v11, i64 %v209
  %v211 = load float, ptr %v210, align 4
  %v212 = mul i64 %v96, 32
  %v213 = add i64 %v212, 16
  %v214 = add i64 %v213, %v26
  %v215 = mul i64 %v214, 128
  %v216 = add i64 %v215, %v18
  %v217 = add i64 %v216, 80
  %v218 = getelementptr inbounds float, ptr %v11, i64 %v217
  %v219 = load float, ptr %v218, align 4
  %v220 = fmul float %v102, %v110
  %v221 = fmul float %v220, %v219
  %v222 = mul i64 %v96, 32
  %v223 = add i64 %v222, %v26
  %v224 = mul i64 %v223, 128
  %v225 = add i64 %v224, %v18
  %v226 = add i64 %v225, 96
  %v227 = getelementptr inbounds float, ptr %v11, i64 %v226
  %v228 = load float, ptr %v227, align 4
  %v229 = mul i64 %v96, 32
  %v230 = add i64 %v229, 16
  %v231 = add i64 %v230, %v26
  %v232 = mul i64 %v231, 128
  %v233 = add i64 %v232, %v18
  %v234 = add i64 %v233, 96
  %v235 = getelementptr inbounds float, ptr %v11, i64 %v234
  %v236 = load float, ptr %v235, align 4
  %v237 = fmul float %v103, %v110
  %v238 = fmul float %v237, %v236
  %v239 = mul i64 %v96, 32
  %v240 = add i64 %v239, %v26
  %v241 = mul i64 %v240, 128
  %v242 = add i64 %v241, %v18
  %v243 = add i64 %v242, 112
  %v244 = getelementptr inbounds float, ptr %v11, i64 %v243
  %v245 = load float, ptr %v244, align 4
  %v246 = mul i64 %v96, 32
  %v247 = add i64 %v246, 16
  %v248 = add i64 %v247, %v26
  %v249 = mul i64 %v248, 128
  %v250 = add i64 %v249, %v18
  %v251 = add i64 %v250, 112
  %v252 = getelementptr inbounds float, ptr %v11, i64 %v251
  %v253 = load float, ptr %v252, align 4
  %v254 = fmul float %v104, %v110
  %v255 = fmul float %v254, %v253
  %v256 = fadd float 0.0, %v136
  %v257 = fadd float %v256, %v170
  %v258 = fadd float %v257, %v204
  %v259 = fadd float %v258, %v238
  %v260 = fadd float 0.0, %v153
  %v261 = fadd float %v260, %v187
  %v262 = fadd float %v261, %v221
  %v263 = fadd float %v262, %v255
  %v264 = mul i64 %v18, 8
  %v265 = add i64 %v264, %v19
  %v266 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_0, i64 %v265
  br label %bb17
bb17:
  store float %v259, ptr addrspace(3) %v266, align 4
  %v267 = add i64 %v18, 16
  %v268 = mul i64 %v267, 8
  %v269 = add i64 %v268, %v19
  %v270 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_0, i64 %v269
  br label %bb18
bb18:
  store float %v263, ptr addrspace(3) %v270, align 4
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb19
bb19:
  %v272 = icmp ult i64 %v17, 32
  %v273 = xor i1 %v272, 1
  br i1 %v273, label %bb32, label %bb20
bb20:
  %v274 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_0, i64 %v265
  br label %bb21
bb21:
  %v275 = load float, ptr addrspace(3) %v274, align 4
  %v276 = add i64 %v18, 4
  %v277 = mul i64 %v276, 8
  %v278 = add i64 %v277, %v19
  %v279 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_0, i64 %v278
  br label %bb22
bb22:
  %v280 = load float, ptr addrspace(3) %v279, align 4
  %v281 = add i64 %v18, 8
  %v282 = mul i64 %v281, 8
  %v283 = add i64 %v282, %v19
  %v284 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_0, i64 %v283
  br label %bb23
bb23:
  %v285 = load float, ptr addrspace(3) %v284, align 4
  %v286 = add i64 %v18, 12
  %v287 = mul i64 %v286, 8
  %v288 = add i64 %v287, %v19
  %v289 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_0, i64 %v288
  br label %bb24
bb24:
  %v290 = load float, ptr addrspace(3) %v289, align 4
  %v291 = add i64 %v18, 16
  %v292 = mul i64 %v291, 8
  %v293 = add i64 %v292, %v19
  %v294 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_0, i64 %v293
  br label %bb25
bb25:
  %v295 = load float, ptr addrspace(3) %v294, align 4
  %v296 = add i64 %v18, 20
  %v297 = mul i64 %v296, 8
  %v298 = add i64 %v297, %v19
  %v299 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_0, i64 %v298
  br label %bb26
bb26:
  %v300 = load float, ptr addrspace(3) %v299, align 4
  %v301 = add i64 %v18, 24
  %v302 = mul i64 %v301, 8
  %v303 = add i64 %v302, %v19
  %v304 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_0, i64 %v303
  br label %bb27
bb27:
  %v305 = load float, ptr addrspace(3) %v304, align 4
  %v306 = add i64 %v18, 28
  %v307 = mul i64 %v306, 8
  %v308 = add i64 %v307, %v19
  %v309 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_0, i64 %v308
  br label %bb28
bb28:
  %v310 = load float, ptr addrspace(3) %v309, align 4
  %v311 = fadd float %v275, %v280
  %v312 = fadd float %v285, %v311
  %v313 = fadd float %v290, %v312
  %v314 = fadd float %v295, %v313
  %v315 = fadd float %v300, %v314
  %v316 = fadd float %v305, %v315
  %v317 = fadd float %v310, %v316
  %v318 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v317, i32 16, i32 31) #0
  br label %bb46
bb29:
  %v319 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_1, i64 %v19
  br label %bb30
bb30:
  store float %v389, ptr addrspace(3) %v319, align 4
  br label %bb31
bb31:
  br label %bb32
bb32:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb33
bb33:
  %v321 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_1, i64 %v19
  br label %bb34
bb34:
  %v322 = load float, ptr addrspace(3) %v321, align 4
  %v323 = fsub float %v121, %v322
  %v324 = fmul float %v323, %v114
  %v325 = fmul float %v134, %v324
  %v326 = fadd float %v135, %v325
  %v327 = fmul float %v326, %v127
  %v328 = fmul float %v151, %v324
  %v329 = fadd float %v152, %v328
  %v330 = fmul float %v329, %v143
  %v331 = fmul float %v168, %v324
  %v332 = fadd float %v169, %v331
  %v333 = fmul float %v332, %v160
  %v334 = fmul float %v185, %v324
  %v335 = fadd float %v186, %v334
  %v336 = fmul float %v335, %v177
  %v337 = fmul float %v202, %v324
  %v338 = fadd float %v203, %v337
  %v339 = fmul float %v338, %v194
  %v340 = fmul float %v219, %v324
  %v341 = fadd float %v220, %v340
  %v342 = fmul float %v341, %v211
  %v343 = fmul float %v236, %v324
  %v344 = fadd float %v237, %v343
  %v345 = fmul float %v344, %v228
  %v346 = fmul float %v253, %v324
  %v347 = fadd float %v254, %v346
  %v348 = fmul float %v347, %v245
  %v349 = fadd float 0.0, %v327
  %v350 = fadd float %v349, %v330
  %v351 = fadd float %v350, %v333
  %v352 = fadd float %v351, %v336
  %v353 = fadd float %v352, %v339
  %v354 = fadd float %v353, %v342
  %v355 = fadd float %v354, %v345
  %v356 = fadd float %v355, %v348
  %v357 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_2, i64 %v265
  br label %bb35
bb35:
  store float %v356, ptr addrspace(3) %v357, align 4
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb36
bb36:
  %v359 = xor i1 %v272, 1
  br i1 %v359, label %bb44, label %bb37
bb37:
  %v360 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_2, i64 %v265
  br label %bb38
bb38:
  %v361 = load float, ptr addrspace(3) %v360, align 4
  %v362 = add i64 %v18, 4
  %v363 = mul i64 %v362, 8
  %v364 = add i64 %v363, %v19
  %v365 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_2, i64 %v364
  br label %bb39
bb39:
  %v366 = load float, ptr addrspace(3) %v365, align 4
  %v367 = add i64 %v18, 8
  %v368 = mul i64 %v367, 8
  %v369 = add i64 %v368, %v19
  %v370 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_2, i64 %v369
  br label %bb40
bb40:
  %v371 = load float, ptr addrspace(3) %v370, align 4
  %v372 = add i64 %v18, 12
  %v373 = mul i64 %v372, 8
  %v374 = add i64 %v373, %v19
  %v375 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_a858d0f5fb2dcc30_2, i64 %v374
  br label %bb41
bb41:
  %v376 = load float, ptr addrspace(3) %v375, align 4
  %v377 = fadd float %v361, %v366
  %v378 = fadd float %v371, %v377
  %v379 = fadd float %v376, %v378
  %v380 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v379, i32 16, i32 31) #0
  br label %bb48
bb42:
  %v381 = mul i64 %v22, %v27
  %v382 = add i64 %v381, %v96
  %v383 = mul i64 %v382, 128
  %v384 = add i64 %v383, %v25
  %v385 = getelementptr inbounds float, ptr %v9, i64 %v384
  store float %v394, ptr %v385, align 4
  br label %bb43
bb43:
  br label %bb44
bb44:
  %v386 = add i64 %v96, 1
  br label %bb15
bb45:
  store float %v97, ptr %v59, align 4
  store float %v98, ptr %v64, align 4
  store float %v99, ptr %v69, align 4
  store float %v100, ptr %v74, align 4
  store float %v101, ptr %v79, align 4
  store float %v102, ptr %v84, align 4
  store float %v103, ptr %v89, align 4
  store float %v104, ptr %v94, align 4
  ret void
bb46:
  %v387 = fadd float %v317, %v318
  %v388 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v387, i32 8, i32 31) #0
  br label %bb47
bb47:
  %v389 = fadd float %v387, %v388
  %v390 = icmp ult i64 %v17, 8
  %v391 = xor i1 %v390, 1
  br i1 %v391, label %bb31, label %bb29
bb48:
  %v392 = fadd float %v379, %v380
  %v393 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v392, i32 8, i32 31) #0
  br label %bb49
bb49:
  %v394 = fadd float %v392, %v393
  %v395 = icmp ult i64 %v17, 8
  %v396 = xor i1 %v395, 1
  br i1 %v396, label %bb43, label %bb42
}


@llvm.used = appending global [1 x ptr] [ptr @gdn_f32_128], section "llvm.metadata"

attributes #0 = { convergent }
