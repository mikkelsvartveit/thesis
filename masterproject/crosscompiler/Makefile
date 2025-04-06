#--------------------------------------------------------------
# Singularity Container Build Makefile
#
# This Makefile manages building Singularity containers:
# - Base images from definition files
# - Cross-compiler images with Slurm support when available
#--------------------------------------------------------------

#--------------------------------------------------------------
# Directory Configuration
#--------------------------------------------------------------
SINGULARITY_DIR       := ./singularity-images
OUTPUT_DIR            := ./output
RESULT_DIR            := ./results
SLURM_LOGS_DIR        := ./slurm-logs
SUBMIT_SCRIPTS_DIR    := $(SINGULARITY_DIR)/submit-scripts
CROSSCOMPILER_DEFS    := $(SINGULARITY_DIR)/crosscompiler-definitions
CROSSCOMPILER_IMAGES  := $(SINGULARITY_DIR)/crosscompiler-images

#--------------------------------------------------------------
# File Lists
#--------------------------------------------------------------
# Base image definition files and their corresponding SIF files
BASE_DEF_FILES        := $(wildcard $(SINGULARITY_DIR)/*.def)
BASE_SIF_FILES        := $(BASE_DEF_FILES:.def=.sif)

#--------------------------------------------------------------
# Phony Targets
#--------------------------------------------------------------
.PHONY: help prepare baseimages baseimagessbatch crosscompilers crosscompilerscripts compileall compileallsbatch dataset clean cleanall

#--------------------------------------------------------------
# Help Targets
#--------------------------------------------------------------
help:
	@echo "Available targets:"
	@echo "  help               - Display this help message"
	@echo "  prepare            - Create directories and extract sources"
	@echo "  baseimages         - Build base Singularity images locally"
	@echo "  baseimagessbatch   - Build base Singularity images using Slurm"
	@echo "  crosscompilerscripts - Generate scripts for crosscompiler images"
	@echo "  crosscompilers     - Build cross-compiler images from list in generate_build_scripts.sh"
	@echo "  compileall         - Build libraries using SIF containers and library list in compile_all_sif.sh"
	@echo "  compileallsbatch   - Build all libraries using Slurm batch jobs"
	@echo "  dataset            - Generate dataset from built libraries"
	@echo "  clean              - Removes intermediate files and output from crosscompilers"
	@echo "  cleanall           - clean, but also removes sif-images"

#--------------------------------------------------------------
# Preparation Target
#--------------------------------------------------------------

# Add this to your directory configuration
PREPARE_STAMP := .prepare_complete

# Modify your prepare target
prepare: $(PREPARE_STAMP)

$(PREPARE_STAMP):
	@echo "Creating directories and extracting sources..."
	@mkdir -p $(OUTPUT_DIR) $(CROSSCOMPILER_DEFS) $(CROSSCOMPILER_IMAGES)
	@mkdir -p $(SUBMIT_SCRIPTS_DIR)
	@mkdir -p $(RESULT_DIR)/library_files $(RESULT_DIR)/text_asm $(RESULT_DIR)/text_bin
	@mkdir -p $(SLURM_LOGS_DIR)/logs $(SLURM_LOGS_DIR)/compilelogs $(SUBMIT_SCRIPTS_DIR)
	@cd sources && tar xf sources.tar.gz && cd ..
	@touch $(PREPARE_STAMP)

#--------------------------------------------------------------
# Base Image Building
#--------------------------------------------------------------
baseimages: $(BASE_SIF_FILES)
	@echo "All base images built successfully"

baseimagessbatch: $(PREPARE_STAMP)
	@echo "Submitting base image build to Slurm..."
	@mkdir -p $(SLURM_LOGS_DIR)/compilelogs
	sbatch \
		--account="share-ie-idi" \
		--time="00:45:00" \
		--nodes="1" \
		--cpus-per-task="16" \
		--mem="16G" \
		--partition="CPUQ" \
		--job-name="base-images" \
		--output="$(SLURM_LOGS_DIR)/compilelogs/base-images-%j.out" \
		--error="$(SLURM_LOGS_DIR)/compilelogs/base-images-%j.err" \
		--wrap="make baseimages"
	@echo "Base image build job submitted to Slurm"

# Rule to build base SIF files from DEF files
$(SINGULARITY_DIR)/%.sif: $(SINGULARITY_DIR)/%.def | $(PREPARE_STAMP)
	@echo "Building base image: $@"
	singularity build --fakeroot $@ $<

#--------------------------------------------------------------
# Cross-compiler Building
#--------------------------------------------------------------
# Generate the build scripts and singularity .def for cross-compilers
crosscompilerscripts: | $(PREPARE_STAMP)
	@echo "Generating build scripts..."
	./scripts/generate_build_scripts.sh

crosscompilers: baseimages crosscompilerscripts 
	$(eval CROSSCOMPILER_DEFS_FILES := $(wildcard $(CROSSCOMPILER_DEFS)/*.def))
	$(eval CROSSCOMPILER_SIF_FILES := $(patsubst $(CROSSCOMPILER_DEFS)/%.def,$(CROSSCOMPILER_IMAGES)/crosscompiler-%.sif,$(CROSSCOMPILER_DEFS_FILES)))
	$(MAKE) $(CROSSCOMPILER_SIF_FILES)
	@echo "All cross-compiler images built or started building"


# Build cross-compiler images using Slurm or locally
$(CROSSCOMPILER_IMAGES)/crosscompiler-%.sif: $(CROSSCOMPILER_DEFS)/%.def | $(PREPARE_STAMP)
	@if [ ! -f "$@" ]; then \
		echo "Building cross-compiler image: $@"; \
		if command -v sbatch >/dev/null 2>&1; then \
			echo "  Using Slurm for building"; \
			sbatch $(SUBMIT_SCRIPTS_DIR)/build_$*.slurm; \
		else \
			echo "  Building locally (Slurm not available)"; \
			chmod +x $(SUBMIT_SCRIPTS_DIR)/build_$*.slurm && $(SUBMIT_SCRIPTS_DIR)/build_$*.slurm; \
		fi; \
	else \
		echo "Using existing cross-compiler image: $@"; \
	fi

#--------------------------------------------------------------
# Library-compiling
#--------------------------------------------------------------
compileall: | $(PREPARE_STAMP)
	@echo "Checking for crosscompiler images..."
	@if ls $(CROSSCOMPILER_IMAGES)/*.sif 1>/dev/null 2>&1; then \
		echo "Starting compilation of libraries"; \
		./scripts/compile_all_sif.sh; \
	else \
		echo "Warning: No crosscompiler images found. Build them first with 'make crosscompilers'"; \
	fi

compileallsbatch: | $(PREPARE_STAMP)
	@echo "Submitting library compilation to Slurm..."
	./scripts/compile_all_sif_sbatch.sh

$(OUTPUT_DIR)/%: $(CROSSCOMPILER_IMAGES)/crosscompiler-%.sif
	@echo "Generating libraries for architecture: $*"
	@mkdir -p $@
	./scripts/compile_all_sif.sh $*

#--------------------------------------------------------------
# Dataset building
#--------------------------------------------------------------
dataset: | $(PREPARE_STAMP)
	@echo "Generating dataset..."
	./scripts/result-gen/generate_dataset.sh

#--------------------------------------------------------------
# Cleanup
#--------------------------------------------------------------
clean:
	@echo "Cleaning up intermediate files and results"
	rm -rf $(OUTPUT_DIR)/*
	rm -rf $(SLURM_LOGS_DIR)/logs/*
	rm -rf $(SLURM_LOGS_DIR)/compilelogs/*
	rm -rf $(SUBMIT_SCRIPTS_DIR)/*
	rm -rf $(RESULT_DIR)/library_files
	rm -rf $(RESULT_DIR)/text_asm
	rm -rf $(RESULT_DIR)/text_bin
	rm -f $(PREPARE_STAMP)


cleanall: clean
	@echo "Cleaning up built images..."
	rm -rf $(SINGULARITY_DIR)/*.sif
	rm -rf $(CROSSCOMPILER_DEFS)/*
	rm -rf $(CROSSCOMPILER_IMAGES)/*