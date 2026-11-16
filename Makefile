# Native run and quality gates (§21.2). Backend through uv, dashboard through Bun.
#
#   make install   install both sides strictly from their lockfiles
#   make weights   download the three pose models into models/ (DEVICE=mps to test on MPS)
#   make run       migrate, then start the API and the dashboard together (Ctrl-C stops both)
#   make check     every backend and frontend gate
#   make camera    bridge the local webcam to rtsp://127.0.0.1:8554/webcam (mediamtx + ffmpeg)
#   make zip       package the project for the TA supervisor into dist/
#   make zip-dataset  package the downloaded datasets (and replay cache) into dist/
#
# The backend reads `backend/.env`, so its commands run from `backend/`.
# macOS and Linux use this file; on Windows use scripts/dev.ps1 (same targets).

HOST ?= 127.0.0.1
PORT ?= 8000

BACKEND  := backend
FRONTEND := frontend

# Webcam bridge for FALL_VIDEO__MODE=rtsp. mediamtx must load a config: with none it starts
# with no paths and rejects the publish with "400 Bad Request". It runs from a temporary
# directory because it writes auto.key/auto.crt into its working directory.
UNAME := $(shell uname -s 2>/dev/null)

# On Linux, an empty MEDIAMTX_CONFIG lets mediamtx search its own default locations
# (./mediamtx.yml, /usr/local/etc, /usr/etc, /etc/mediamtx).
ifeq ($(UNAME),Darwin)
MEDIAMTX_CONFIG ?= $(shell brew --prefix 2>/dev/null)/etc/mediamtx/mediamtx.yml
else
MEDIAMTX_CONFIG ?=
endif
MEDIAMTX_DIR    ?= $(or $(TMPDIR),/tmp)
RTSP_PORT       ?= 8554
RTSP_URL        ?= rtsp://127.0.0.1:$(RTSP_PORT)/webcam
WEBCAM_DEVICE   ?= 0
WEBCAM_SIZE     ?= 1280x720
WEBCAM_FPS      ?= 30

# Capture input per OS: AVFoundation (macOS) takes an index or name, V4L2 (Linux) an
# index N for /dev/videoN.
ifeq ($(UNAME),Darwin)
WEBCAM_INPUT = -f avfoundation -framerate $(WEBCAM_FPS) -video_size $(WEBCAM_SIZE) \
  -pixel_format nv12 -i $(WEBCAM_DEVICE):none
else
WEBCAM_INPUT = -f v4l2 -framerate $(WEBCAM_FPS) -video_size $(WEBCAM_SIZE) \
  -i /dev/video$(WEBCAM_DEVICE)
endif

.DEFAULT_GOAL := help
.PHONY: help install weights migrate run backend frontend rtsp webcam camera test lint typecheck check build ta-docx ta-docx-status zip zip-dataset

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install: ## Install backend and frontend dependencies from the lockfiles
	cd $(BACKEND) && uv sync --locked
	cd $(FRONTEND) && bun install --frozen-lockfile

weights: ## Download and verify the YOLOv8n/YOLO11n/YOLO26n pose weights into models/
	uv run --project $(BACKEND) python scripts/phase0_ml_smoke.py --device $(or $(DEVICE),cpu)

migrate: ## Apply database migrations
	cd $(BACKEND) && uv run alembic upgrade head

run: migrate ## Start the API and the dashboard together
	@echo "API       http://$(HOST):$(PORT)"
	@echo "Dashboard http://localhost:5173"
	@trap 'kill 0' INT TERM EXIT; \
	(cd $(BACKEND) && uv run uvicorn fall_detection.api.app:app --host $(HOST) --port $(PORT) --workers 1) & \
	(cd $(FRONTEND) && FALL_API_TARGET=http://$(HOST):$(PORT) bun run dev) & \
	wait

backend: migrate ## Start only the API
	cd $(BACKEND) && uv run uvicorn fall_detection.api.app:app --host $(HOST) --port $(PORT) --workers 1

frontend: ## Start only the dashboard dev server
	cd $(FRONTEND) && FALL_API_TARGET=http://$(HOST):$(PORT) bun run dev

rtsp: ## Start only the mediamtx RTSP server
	cd $(MEDIAMTX_DIR) && mediamtx $(MEDIAMTX_CONFIG)

webcam: ## Publish the webcam over RTSP (needs `make rtsp` running)
	@until nc -z 127.0.0.1 $(RTSP_PORT) 2>/dev/null; do sleep 0.2; done
	ffmpeg -hide_banner -loglevel warning $(WEBCAM_INPUT) \
	  -c:v libx264 -preset ultrafast -tune zerolatency -g $(WEBCAM_FPS) -pix_fmt yuv420p \
	  -f rtsp -rtsp_transport tcp $(RTSP_URL)

camera: ## Start the RTSP server and the webcam publisher together
	@trap 'kill 0' INT TERM EXIT; \
	$(MAKE) --no-print-directory rtsp & \
	$(MAKE) --no-print-directory webcam & \
	wait

test: ## Run backend and frontend tests
	cd $(BACKEND) && uv run pytest
	cd $(FRONTEND) && bun run test

lint: ## Lint and format-check both sides
	cd $(BACKEND) && uv run ruff check . && uv run ruff format --check .
	cd $(FRONTEND) && bun run lint

typecheck: ## Type-check both sides
	cd $(BACKEND) && uv run basedpyright
	cd $(FRONTEND) && bun run typecheck

check: lint typecheck test ## Every gate, including the migration drift check
	cd $(BACKEND) && uv run alembic check

build: ## Build the dashboard for production
	cd $(FRONTEND) && bun run build

ta-docx: ## Rebuild docs/ta/laporan-ta.docx from the Typst sources at REF (default HEAD)
	docs/ta/word/build.sh $(or $(REF),HEAD)

ta-docx-status: ## Show Typst commits not yet in docs/ta/laporan-ta.docx
	@base=$$(sed -n 's/^commit: //p' docs/ta/laporan-ta.docx.source); \
	echo "laporan-ta.docx dibangun dari $$(git rev-parse --short $$base)"; \
	changes=$$(git log --oneline $$base..HEAD -- docs/ta ':!docs/ta/laporan-ta.docx*' ':!docs/ta/word'); \
	if [ -z "$$changes" ]; then echo "Sudah sesuai dengan Typst di HEAD."; \
	else echo "Commit Typst sesudahnya:"; echo "$$changes"; fi

# Supervisor package: every file git tracks or would track (so .gitignore already drops
# datasets/raw, datasets/cache, models/, .venv, node_modules, experiments/runs, .env, caches),
# minus docs/, THESIS_PLAN.md, .gitignore files and the AI-assistant notes.
# The commit is recorded three ways: in the file name, in COMMIT.txt at the zip root, and in
# the zip comment (`unzip -z`). Uncommitted changes are packaged too and marked "-dirty".
ZIP_NAME    ?= fall-detection-ta
ZIP_EXCLUDE := docs/ THESIS_PLAN.md AGENTS.md CLAUDE.md anti-slop/ graphify-out/ .vscode/

zip: ## Package the project for the TA supervisor into dist/<name>-<date>-<commit>.zip
	@set -e; \
	commit=$$(git rev-parse HEAD); short=$$(git rev-parse --short HEAD); \
	dirty=$$(git status --porcelain | grep -q . && echo -dirty || true); \
	out="$(CURDIR)/dist/$(ZIP_NAME)-$$(date +%Y%m%d)-$$short$$dirty.zip"; \
	stage=$$(mktemp -d); trap 'rm -rf "$$stage"' EXIT; \
	git ls-files -co --exclude-standard \
	  | grep -v -e '/\._' -e '^\._' -e '\(^\|/\)\.gitignore$$' $(foreach x,$(ZIP_EXCLUDE),-e '^$(x)') \
	  | while IFS= read -r f; do [ -f "$$f" ] && echo "$$f"; done > "$$stage/files.txt"; \
	rsync -a --files-from="$$stage/files.txt" . "$$stage/$(ZIP_NAME)/"; \
	{ echo "commit: $$commit$$dirty"; \
	  echo "branch: $$(git rev-parse --abbrev-ref HEAD)"; \
	  echo "date:   $$(git log -1 --format=%cI HEAD)"; \
	  echo "subject: $$(git log -1 --format=%s HEAD)"; \
	  echo "packed: $$(date +%Y-%m-%dT%H:%M:%S%z)"; \
	  [ -z "$$dirty" ] || echo "note:   berisi perubahan yang belum di-commit"; \
	} > "$$stage/$(ZIP_NAME)/COMMIT.txt"; \
	mkdir -p dist; rm -f "$$out"; \
	(cd "$$stage" && COPYFILE_DISABLE=1 zip -qrX "$$out" "$(ZIP_NAME)"); \
	echo "$(ZIP_NAME) $$commit$$dirty" | zip -qz "$$out"; \
	[ -z "$$dirty" ] || echo "Peringatan: ada perubahan yang belum di-commit; zip ditandai -dirty."; \
	echo "$$(wc -l < "$$stage/files.txt" | tr -d ' ') file -> $$out ($$(du -h "$$out" | cut -f1))"

# Dataset package, so the supervisor does not have to download again. It uses the same root
# folder as `make zip`, so extracting both zips in one place gives a working tree. Already
# compressed media (.zip/.avi/.mp4) is stored, not recompressed. COCO=1 adds the COCO-pose
# val set that pose_benchmark.py uses (~1.8 GB); it is downloaded by Ultralytics otherwise.
DATASET_PARTS := datasets/raw/urfd datasets/raw/caucafall datasets/raw/demo \
                 datasets/cache/fall_observations \
                 $(if $(filter 1,$(COCO)),datasets/raw/ultralytics datasets/cache/coco-pose-val.yaml)

zip-dataset: ## Package datasets/raw (URFD, CAUCAFall) + replay cache into dist/ (COCO=1 adds COCO-pose)
	@set -e; \
	for d in $(DATASET_PARTS); do [ -e "$$d" ] || { echo "Tidak ada: $$d" >&2; exit 2; }; done; \
	out="$(CURDIR)/dist/$(ZIP_NAME)-dataset-$$(date +%Y%m%d)$(if $(filter 1,$(COCO)),-coco).zip"; \
	stage=$$(mktemp -d); trap 'rm -rf "$$stage"' EXIT; \
	ln -s "$(CURDIR)" "$$stage/$(ZIP_NAME)"; \
	{ echo "Dataset untuk $(ZIP_NAME); ekstrak di folder yang sama dengan zip kode."; \
	  echo "packed: $$(date +%Y-%m-%dT%H:%M:%S%z)"; \
	  echo "commit: $$(git rev-parse HEAD) (kode saat dataset dikemas)"; echo; \
	  for d in $(DATASET_PARTS); do printf '%-40s %6s file  %s\n' "$$d" \
	    "$$(find "$$d" -type f ! -name '._*' ! -name .DS_Store | wc -l | tr -d ' ')" \
	    "$$(du -sh "$$d" | cut -f1)"; done; \
	} > "$$stage/DATASET.txt"; \
	mkdir -p "$$stage/meta/$(ZIP_NAME)/datasets"; mv "$$stage/DATASET.txt" "$$stage/meta/$(ZIP_NAME)/datasets/"; \
	mkdir -p dist; rm -f "$$out"; \
	echo "Mengemas $(DATASET_PARTS) ..."; \
	(cd "$$stage" && COPYFILE_DISABLE=1 zip -qrX -n .zip:.avi:.mp4 "$$out" \
	  $(addprefix $(ZIP_NAME)/,$(DATASET_PARTS)) -x '*/._*' '*/.DS_Store'); \
	(cd "$$stage/meta" && zip -qX "$$out" $(ZIP_NAME)/datasets/DATASET.txt); \
	echo "$$(unzip -Z1 "$$out" | grep -vc '/$$') file -> $$out ($$(du -h "$$out" | cut -f1))"
