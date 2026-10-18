"""Compile one pinned FA4 target offline; called by regenerate.py, never by serving."""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,os,re,sys,time,traceback
ap=argparse.ArgumentParser()
ap.add_argument("--target",required=True,choices=["sm120","sm90a","sm100a-1cta","sm100a-2cta","sm103a-1cta","sm103a-2cta"])
ap.add_argument("--source-tree",type=Path,required=True)
ap.add_argument("--sdk-site",type=Path,required=True)
ap.add_argument("--output",type=Path,required=True)
args=ap.parse_args();target=args.target
arch={"sm120":"sm_120","sm90a":"sm_90a"}.get(target,"sm_"+target.split("-")[0][2:])
out=args.output.resolve();out.mkdir(parents=True)
os.environ.update(PYTHONDONTWRITEBYTECODE="1",CUTE_DSL_ARCH=arch,CUTE_DSL_KEEP="ptx,cubin",CUTE_DSL_DUMP_DIR=str(out),CUDA_CACHE_PATH=str(out/"cuda-cache"))
sys.path[:0]=[str(args.source_tree.resolve()),str(args.sdk_site.resolve()),str(args.sdk_site.resolve()/"nvidia_cutlass_dsl/dsl_packages")]
assert importlib.metadata.version("nvidia-cutlass-dsl")=="4.8.0"
rec={"completed":False,"GPU_execution":False,"started_unix":time.time(),"architecture":arch,"candidate":target,"source_official_commit":"e9d8944c855489e03c66262d07ba4c81d9d8dd31","on_hardware_validated":False,"quality_accepted":False}
try:
 import cutlass,cutlass.cute as cute
 from cuda.bindings import driver as cuda
 from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120
 from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
 from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
 if target=="sm120":
  FN=FlashAttentionForwardSm120(cutlass.BFloat16,256,256,6,is_causal=True,pack_gqa=True,tile_m=64,tile_n=64,num_stages=1,num_threads=128,Q_in_regs=False)
  config={"tile":[64,64],"threads":128,"cluster":[1,1,1],"pack_gqa":True,"features":["warp-MMA","cp.async"]}
 elif target=="sm90a":
  FN=FlashAttentionForwardSm90(cutlass.BFloat16,256,256,6,is_causal=True,pack_gqa=True,tile_m=128,tile_n=64,num_stages=2,num_threads=384,Q_in_regs=False,intra_wg_overlap=True,mma_pv_is_rs=True)
  config={"tile":[128,64],"threads":384,"cluster":[1,1,1],"pack_gqa":True,"features":["WGMMA","TMA","warp-specialization"]}
 else:
  two=target.endswith("2cta");pack=not two
  FN=FlashAttentionForwardSm100(256,256,6,is_causal=True,pack_gqa=pack,m_block_size=128,n_block_size=128,q_stage=1,is_static_persistent=True,use_2cta_instrs=two)
  config={"tile":[128,128],"threads":"official-kernel","cluster":[2 if two else 1,1,1],"pack_gqa":pack,"features":["tcgen05","TMEM","TMA"]+(["2-CTA-MMA"] if two else [])}
 @cute.jit
 def entry(q:cute.Tensor,k:cute.Tensor,v:cute.Tensor,o:cute.Tensor,usedq:cute.Tensor,usedk:cute.Tensor,scale:cutlass.Float32,stream:cuda.CUstream):
  FN(q,k,v,o,None,scale,mSeqUsedQ=usedq,mSeqUsedK=usedk,stream=stream)
 shape=(1,cute.sym_int32(divisibility=64),24,256)
 def make(shape,dtype=cutlass.BFloat16,align=16):return cute.runtime.make_fake_compact_tensor(dtype,shape,stride_order=tuple(reversed(range(len(shape)))),assumed_align=align)
 q=make(shape);k=make((1,cute.sym_int32(divisibility=32),4,256));v=make((1,cute.sym_int32(divisibility=32),4,256));o=make(shape)
 usedq=make((1,),cutlass.Int32,4);usedk=make((1,),cutlass.Int32,4);stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False)
 compiled=cute.compile(entry,q,k,v,o,usedq,usedk,cutlass.Float32(.0625),stream,options="--gpu-arch "+arch+" --keep-ptx --keep-cubin")
 rec["config"]=config
 for name,content in [("attention.ptx",compiled.__ptx__),("attention.cubin",compiled.__cubin__)]:
  if isinstance(content,str) and name.endswith("cubin") and Path(content).exists():content=Path(content).read_bytes()
  assert content is not None,name
  if isinstance(content,str):content=content.encode()
  (out/name).write_bytes(content);rec[name]={"sha256":hashlib.sha256(content).hexdigest(),"bytes":len(content)}
 ptx=(out/"attention.ptx").read_text()
 rec["kernel_symbols"]=list(compiled.kernel_info.keys())
 rec["PTX_features"]={name:bool(re.search(regex,ptx)) for name,regex in {"warp_MMA":r"mma\.sync","WGMMA":r"wgmma\.mma_async","TMA":r"cp\.async\.bulk\.tensor","tcgen05":r"tcgen05\.mma","TMEM":r"tcgen05\.(alloc|ld|st)","two_CTA":r"cta_group::2"}.items()}
 compiled.export_to_c(str(out),"attention_host",function_prefix="infer_fa4_"+target.replace("-","_"))
 rec["sdk_version"]=importlib.metadata.version("nvidia-cutlass-dsl")
 assert rec["sdk_version"]=="4.8.0"
 rec["completed"]=True
except BaseException as e:rec.update(error=repr(e),traceback=traceback.format_exc());raise
finally:
 rec["finished_unix"]=time.time()
 with (out/"result.json").open("x") as f:json.dump(rec,f,indent=2)
 print(json.dumps({k:v for k,v in rec.items() if k not in ["traceback","kernel_symbols"]}),flush=True)
