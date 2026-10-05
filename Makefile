.PHONY: check check-rust check-tools check-security check-metal check-cuda check-msrv check-cpu check-linux check-linux-numa clean-artifacts

clean-artifacts:
	python3 tools/check/clean-artifacts.py

check:
	./tools/check/gate.sh all

check-rust:
	./tools/check/gate.sh rust

check-tools:
	./tools/check/gate.sh tools

check-security:
	./tools/check/gate.sh security

check-cuda:
	./tools/check/gate.sh cuda

check-metal:
	./tools/check/gate.sh metal

check-msrv:
	./tools/check/gate.sh msrv

check-cpu:
	./tools/check/gate.sh cpu

check-linux:
	./tools/check/gate.sh linux

check-linux-numa:
	./tools/check/gate.sh linux-numa
