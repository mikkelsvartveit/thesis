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
RESULT_DIR						:= ./results
SLURM_SCRIPTS_DIR     := ./slurm-scripts
CROSSCOMPILER_DEFS    := $(SINGULARITY_DIR)/crosscompiler-definitions
CROSSCOMPILER_IMAGES  := $(SINGULARITY_DIR)/crosscompiler-images

#--------------------------------------------------------------
# File Lists
#--------------------------------------------------------------
# Base image definition files and their corresponding SIF files
BASE_DEF_FILES        := $(wildcard $(SINGULARITY_DIR)/*.def)
BASE_SIF_FILES        := $(BASE_DEF_FILES:.def=.sif)

#--------------------------------------------------------------
# Environment Detection
#--------------------------------------------------------------
# Check if Slurm is available for job submission
SBATCH_AVAILABLE      := $(shell which sbatch 2>/dev/null)

#--------------------------------------------------------------
# Phony Targets
#--------------------------------------------------------------
.PHONY: all help prepare baseimages crosscompilers clean build_scripts build_all build_all_sbatch dataset

#--------------------------------------------------------------
# Help Targets
#--------------------------------------------------------------
help:
	@echo "Available targets:"
	@echo "  help           - Display this help message"
	@echo "  prepare        - Create directories and extract sources"
	@echo "  baseimages     - Build base Singularity images needed for crosscompiler images"
	@echo "  crosscompilers - Build cross-compiler images from list in generate_build_scripts.sh"
	@echo "  build_all      - Build libraries using SIF containers and library list in build-all-sif.sh"
	@echo "  build_all_sbatch - Build all libraries using Slurm batch jobs"
	@echo "  dataset        - Generate dataset from built libraries"
	@echo "  clean          - Removes intermediate image build files and output from crosscompilers"
	@echo "  cleanall       - clean, but also removes sif-images"

#--------------------------------------------------------------
# Preparation Target
#--------------------------------------------------------------
prepare:
	@echo "Creating directories and extracting sources..."
	@mkdir -p $(OUTPUT_DIR) $(CROSSCOMPILER_DEFS) $(CROSSCOMPILER_IMAGES) 
	@mkdir -p $(RESULT_DIR)/library_files $(RESULT_DIR)/text_asm $(RESULT_DIR)/text_bin
	@cd sources && tar xf sources.tar.gz && cd ..

#--------------------------------------------------------------
# Base Image Building
#--------------------------------------------------------------
baseimages: $(BASE_SIF_FILES)
	@echo "All base images built successfully"

# Rule to build base SIF files from DEF files
$(SINGULARITY_DIR)/%.sif: $(SINGULARITY_DIR)/%.def
	@echo "Building base image: $@"
	singularity build --fakeroot $@ $<

#--------------------------------------------------------------
# Cross-compiler Building
#--------------------------------------------------------------
# Generate the build scripts an singularity .def for cross-compilers
build_scripts:
	@echo "Generating build scripts..."
	./scripts/generate_build_scripts.sh

crosscompilers: baseimages build_scripts 
	$(eval CROSSCOMPILER_DEFS_FILES := $(wildcard $(CROSSCOMPILER_DEFS)/*.def))
	$(eval CROSSCOMPILER_SIF_FILES := $(patsubst $(CROSSCOMPILER_DEFS)/%.def,$(CROSSCOMPILER_IMAGES)/crosscompiler-%.sif,$(CROSSCOMPILER_DEFS_FILES)))
	$(MAKE) $(CROSSCOMPILER_SIF_FILES)
	@echo "All cross-compiler images built or started building"


# Build cross-compiler images using Slurm or locally
$(CROSSCOMPILER_IMAGES)/crosscompiler-%.sif: build_scripts
	@if [ ! -f "$@" ]; then \
		echo "Building cross-compiler image: $@"; \
		if command -v sbatch >/dev/null 2>&1; then \
			echo "  Using Slurm for building"; \
			sbatch $(SLURM_SCRIPTS_DIR)/submitscripts/build_$*.slurm; \
		else \
			echo "  Building locally (Slurm not available)"; \
			chmod +x $(SLURM_SCRIPTS_DIR)/submitscripts/build_$*.slurm && $(SLURM_SCRIPTS_DIR)/submitscripts/build_$*.slurm; \
		fi; \
	else \
		echo "Using existing cross-compiler image: $@"; \
	fi

#--------------------------------------------------------------
# Library-building
#--------------------------------------------------------------

build_all: 
	./scripts/build-all-sif.sh

build_all_sbatch:
	./scripts/build-all-sif-sbatch.sh

$(OUTPUT_DIR)/%: $(CROSSCOMPILER_IMAGES)/crosscompiler-%.sif
	@echo "Generating libraries for architecture: $*"
	@mkdir -p $@
	./scripts/build-all-sif.sh $*

#--------------------------------------------------------------
# make dataset
#--------------------------------------------------------------

dataset:
	./scripts/result_gen/generate_dataset.sh

#--------------------------------------------------------------
# Cleanup
#--------------------------------------------------------------
clean:
	@echo "Cleaning up intermediate files and results"
	rm -rf $(OUTPUT_DIR)/*
	rm -rf ./slurm-scripts/logs
	rm -rf ./slurm-scripts/submitscripts
	rm -rf $(CROSSCOMPILER_DEFS)
	rm -rf $(RESULT_DIR)/library_files
	rm -rf $(RESULT_DIR)/text_asm
	rm -rf $(RESULT_DIR)/text_bin


cleanall: clean
	@echo "Cleaning up built images..."
	rm -rf $(SINGULARITY_DIR)/*.sif
	rm -rf $(CROSSCOMPILER_IMAGES)
