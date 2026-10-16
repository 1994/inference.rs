"""Verify embedded AOT identities, host ABI and actual architecture instruction families."""
from pathlib import Path
import hashlib,json,re,struct,subprocess
ROOT=Path(__file__).resolve().parent
EXPECTED={"sm120":("sm_120",["warp_MMA"],1),"sm90a":("sm_90a",["WGMMA","TMA"],1),"sm100a-1cta":("sm_100a",["tcgen05","TMEM","TMA"],1),"sm100a-2cta":("sm_100a",["tcgen05","TMEM","TMA","two_CTA"],2),"sm103a-1cta":("sm_103a",["tcgen05","TMEM","TMA"],1),"sm103a-2cta":("sm_103a",["tcgen05","TMEM","TMA","two_CTA"],2)}
PATTERNS={"warp_MMA":r"mma\.sync","WGMMA":r"wgmma\.mma_async","TMA":r"cp\.async\.bulk\.tensor","tcgen05":r"tcgen05\.mma","TMEM":r"tcgen05\.(alloc|ld|st)","two_CTA":r"cta_group::2"}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 m=json.loads((ROOT/"manifest.json").read_text())
 assert m["schema"]==1 and m["host_ABI"]==1 and not m["runtime_python"]
 assert m["host_platform"]=="linux-x86_64" and m["cutlass_dsl"]=="4.8.0"
 assert m["upstream_commit"]==json.loads((ROOT/"upstream-files.json").read_text())["commit"]
 assert set(e["name"] for e in m["entries"])==set(EXPECTED) and len(m["entries"])==len(EXPECTED)
 for file,digest in m["common_files"].items():assert sha(ROOT/file)==digest,file
 for e in m["entries"]:
  root=ROOT/e["name"];arch,features,cluster=EXPECTED[e["name"]]
  assert e["architecture"]==arch and e["config"]["cluster"]==[cluster,1,1]
  assert e["shape"]=={"query_heads":24,"kv_heads":4,"head_dim":256,"lanes_multiple":64,"context_dynamic":True}
  assert e["profile"]=="bf16-inputs-f32-accumulator-bf16-output"
  for file,digest in e["files"].items():assert sha(root/file)==digest,(root,file)
  ptx=(root/"attention.ptx").read_text()
  actual={name:bool(re.search(regex,ptx)) for name,regex in PATTERNS.items()}
  assert actual==e["features"] and all(actual[f] for f in features)
  assert actual["two_CTA"]==(cluster==2)
  assert re.search(r"\.target\s+"+arch+r"\b",ptx)
  image=(root/"attention.so").read_bytes()
  assert image[:6]==b"\x7fELF\x02\x01" and struct.unpack_from("<H",image,18)[0]==62
  for symbol in [b"infer_fa4_initialize",b"infer_fa4_forward"]:assert symbol+b"\0" in image
  dynamic=subprocess.check_output(["readelf","-d",str(root/"attention.so")],text=True,env={"LC_ALL":"C","PATH":"/usr/bin:/bin"})
  deps=re.findall(r"\(NEEDED\).*?\[([^\]]+)\]",dynamic)
  assert set(deps)<=set(m["runtime_dependencies"]) and {"libcudart.so.13","libcuda.so.1"}<=set(deps),deps
  assert not any("python" in x.lower() or "CuteDSL" in x for x in deps)
  assert not e["task_quality_accepted"],"Model acceptance requires an explicit reviewed qualification receipt"
 print("FA4 AOT: six artifacts, architecture instructions, host ABI, source identity and CUDA runtime dependencies verified")
if __name__=="__main__":main()
