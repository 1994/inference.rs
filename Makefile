# Zig cross-platform builds. See docs/guides/packaging.md for artifacts and GPU acceptance.
.DEFAULT_GOAL := help
PYTHON ?= python3
CARGO ?= cargo
TARGET ?=
CARGO_ZIGBUILD_VERSION := 0.23.4
DIST_DIR ?= artifacts/packages
PACKAGE_FILE ?=
MODEL ?=
GOLDEN ?=

.PHONY: help build local-build test package verify-package accept clean-artifacts
help:
	@echo 'Zig target pipelines (Linux/CUDA or macOS/Metal):'
	@echo '  make build | test | package TARGET=<rust-target>[.<glibc>]'
	@echo '  make local-build  # native release CLI with the platform backend'
	@echo '  make package-linux-cuda | package-macos-metal TARGET=<target>'
	@echo '  make setup-build  # install cargo-zigbuild; requires Zig'
	@echo '  make verify-package PACKAGE_FILE=/path/to/archive.tar.gz'
	@echo '  make accept PACKAGE_FILE=... MODEL=... GOLDEN=...'
	@echo '  DIST_DIR=artifacts/packages overrides the artifact directory.'
	@echo 'Quality gates: check, check-rust, check-tools, check-security,'
	@echo '  check-cuda, check-metal, check-attention, check-msrv, check-cpu,'
	@echo '  check-linux, check-linux-numa, check-package'

build test package:
	$(PYTHON) tools/package/package.py $@ --platform auto --target "$(TARGET)" --out "$(DIST_DIR)"

# Cargo features are compile-time and cannot be selected by the CLI at runtime. Keep the
# packaging pipeline above unchanged, but provide a native developer entry point that selects the
# production backend from the host platform.
local-build:
	@case "$$(uname -s)" in \
		Linux) $(CARGO) build --locked --release -p infer-cli --features cuda ;; \
		Darwin) $(CARGO) build --locked --release -p infer-cli ;; \
		*) echo "unsupported host platform: $$(uname -s)" >&2; exit 2 ;; \
	esac

# Recursive Make expansion is deliberately avoided: each package action sequences
# target checks -> Zig release -> archive -> integrity/native smoke -> publication.
define platform_targets
.PHONY: build-$(1) test-$(1) package-$(1)
build-$(1) test-$(1) package-$(1):
	$$(PYTHON) tools/package/package.py $$(word 1,$$(subst -, ,$$@)) --platform $(1) --target "$$(TARGET)" --out "$$(DIST_DIR)"
endef
$(eval $(call platform_targets,linux-cuda))
$(eval $(call platform_targets,macos-metal))

verify-package:
	$(PYTHON) tools/package/package.py verify --archive "$(PACKAGE_FILE)"

accept:
	$(PYTHON) tools/package/package.py accept --archive "$(PACKAGE_FILE)" --model "$(MODEL)" --golden "$(GOLDEN)"

.PHONY: check check-rust check-tools check-security check-metal check-cuda check-attention check-msrv check-cpu check-linux check-linux-numa check-package
check:
	./tools/check/gate.sh all

check-rust check-tools check-security check-metal check-cuda check-attention check-msrv check-cpu check-linux check-linux-numa:
	./tools/check/gate.sh $(patsubst check-%,%,$@)

check-package:
	$(PYTHON) -m unittest discover -s tools/package -p 'test_*.py'

clean-artifacts:
	$(PYTHON) tools/check/clean-artifacts.py

.PHONY: setup-build
setup-build:
	cargo install --locked cargo-zigbuild --version $(CARGO_ZIGBUILD_VERSION)
	"$${CARGO_ZIGBUILD_ZIG_PATH:-zig}" version
