"""CPU verification of the separate F32 source-storage bridge and its fixed launch ABI."""
from pathlib import Path
import argparse,hashlib,json,re
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--generated",type=Path);args=ap.parse_args()
 p=json.loads((ROOT/"provenance.json").read_text())
 assert p["completed"] and not p["GPU_execution"] and not p["hardware_validation"] and not p["task_quality_accepted"]
 assert p["source_sha256"]==sha(ROOT/"bridge.cu") and p["ptx_sha256"]==sha(ROOT/"bridge.ptx")
 text=(ROOT/"bridge.ptx").read_text();assert re.search(r"\.target\s+sm_90\b",text)
 header=text.split(".entry prepare_fa4_f32(",1)[1].split(")",1)[0]
 assert header.count(".param")==p["argument_count"]==13
 assert header.count(".u64")==9 and header.count(".u32")==2 and header.count(".f32")==2
 if args.generated:assert args.generated.read_bytes()==(ROOT/"bridge.ptx").read_bytes()
 print("F32-cache bridge: source/PTX identity, SM90 baseline and 13-argument ABI verified; hardware/task acceptance pending")
if __name__=="__main__":main()
