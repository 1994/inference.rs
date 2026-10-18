"""Regenerate all six AOT targets using pinned source/SDK; no GPU execution."""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parent
TARGETS=["sm120","sm90a","sm100a-1cta","sm100a-2cta","sm103a-1cta","sm103a-2cta"]
FILES=["attention.ptx","attention.cubin","attention_host.h","attention_host.o","host.cpp","attention.so"]
COMMON=["bridge.cu","bridge.ptx","host.cpp.in","LICENSE.upstream","upstream-files.json","compile.py","regenerate.py","verify-artifact.py"]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(argv,log):
 p=subprocess.run(list(map(str,argv)),capture_output=True,text=True,env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","MAX_JOBS":"1","NVCC_THREADS":"1"})
 log.write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError(f"{argv[0]} failed ({p.returncode}); see {log}")
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--upstream",type=Path,required=True,help="checkout containing the pinned flash_attn/cute tree")
 ap.add_argument("--sdk-site",type=Path,required=True,help="site-packages with nvidia-cutlass-dsl 4.8.0 and its libraries")
 ap.add_argument("--cuda-root",type=Path,default=Path("/opt/cuda"))
 ap.add_argument("--cxx",default="g++")
 ap.add_argument("--output",type=Path,required=True,help="new directory; existing artifacts are never overwritten")
 args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True)
 source=json.loads((ROOT/"upstream-files.json").read_text())
 version=subprocess.check_output([str(args.cuda_root/"bin/nvcc"),"--version"],text=True)
 assert "V13.4.92" in version,"Use pinned CUDA 13.4.92"
 with tempfile.TemporaryDirectory(prefix="infer-fa4-source-") as temporary:
  tree=Path(temporary)
  for item in source["files"]:
   original=args.upstream/item["path"];data=original.read_bytes()
   assert hashlib.sha256(data).hexdigest()==item["sha256"],original
   dest=tree/item["path"];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
  # Isolate CuTe from the parent package's optional FA2 native extension.
  (tree/"flash_attn/__init__.py").write_text("# AOT compilation namespace for verified CuTe sources.\n")
  entries=[]
  for target in TARGETS:
   target_out=out/target
   run([sys.executable,ROOT/"compile.py","--target",target,"--source-tree",tree,"--sdk-site",args.sdk_site,"--output",target_out],out/(target+"-compile.log"))
   result=json.loads((target_out/"result.json").read_text());assert result["completed"]
   prefix="infer_fa4_"+target.replace("-","_")
   (target_out/"host.cpp").write_text((ROOT/"host.cpp.in").read_text().replace("PREFIX",prefix))
   run([args.cxx,"-shared","-fPIC","-O2","-fvisibility=hidden","-I"+str(args.cuda_root/"include"),target_out/"host.cpp",target_out/"attention_host.o","-L"+str(args.cuda_root/"lib64"),"-L"+str(args.cuda_root/"lib64/stubs"),"-lcudart","-lcuda","-Wl,-z,defs","-Wl,-z,relro,-z,now","-Wl,-rpath,$ORIGIN","-o",target_out/"attention.so"],out/(target+"-link.log"))
   entries.append({"name":target,"architecture":result["architecture"],"profile":"bf16-inputs-f32-accumulator-bf16-output","shape":{"query_heads":24,"kv_heads":4,"head_dim":256,"lanes_multiple":64,"context_dynamic":True},"config":result["config"],"features":result["PTX_features"],"files":{f:sha(target_out/f) for f in FILES},"on_hardware_validated":False,"task_quality_accepted":False})
   shutil.copyfile(target_out/"result.json",target_out/"provenance.json")
 for name in COMMON:
  if name!="bridge.ptx":shutil.copyfile(ROOT/name,out/name)
 run([args.cuda_root/"bin/nvcc","--ptx","-arch=compute_90","-O3","--fmad=false","--ftz=false","--prec-div=true",out/"bridge.cu","-o",out/"bridge.ptx"],out/"bridge-compile.log")
 manifest={"schema":1,"upstream_commit":source["commit"],"cutlass_dsl":"4.8.0","cuda_compiler":"13.4.92","host_platform":"linux-x86_64","host_ABI":1,"runtime_python":False,"runtime_dependencies":["libcudart.so.13","libcuda.so.1","libstdc++.so.6","libm.so.6","libgcc_s.so.1","libc.so.6"],"common_files":{f:sha(out/f) for f in COMMON},"entries":entries}
 (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 run([sys.executable,out/"verify-artifact.py"],out/"verify.log")
 print(out)
if __name__=="__main__":main()
