#!/usr/bin/env python3

# Mapping between corpus files and standard architecture names
rename_map = {
    # X86 Family
    "X86.corpus": "i386.corpus",
    "X86-64.corpus": "amd64.corpus",
    # ARM Family
    "ARM64.corpus": "arm64.corpus",
    "ARMel.corpus": "armel.corpus",
    "ARMhf.corpus": "armhf.corpus",
    # MIPS Family
    "MIPSeb.corpus": "mips.corpus",  # double check ✅
    "MIPSel.corpus": "mipsel.corpus",  # double check ✅
    # PowerPC Family
    "PPCeb.corpus": "powerpc.corpus",  # Big endian PPC  #double check ✅
    "PPCel.corpus": "ppc64el.corpus",  # Little endian PPC64 #double check ✅
    # Other major architectures
    "Alpha.corpus": "alpha.corpus",
    "HP-PA.corpus": "hppa.corpus",
    "IA-64.corpus": "ia64.corpus",
    "M68k.corpus": "m68k.corpus",
    "RISC-V.corpus": "riscv64.corpus",
    "S-390.corpus": "s390x.corpus",  # double check vs s390 and s390x ✅
    "SuperH.corpus": "sh4.corpus",  # double check, but need to check endianness as it could be bi
    "SPARC.corpus": "sparc32&64.corpus",  # double check ✅
}

# Files to remove (only bytecode and non-ISA specific)
files_to_remove = [
    # Bytecode formats
    "CUDA.corpus",  # NVIDIA PTX bytecode
    "OCaml.corpus",  # OCaml bytecode
    "WASM.corpus",  # WebAssembly bytecode
    # Special variants that aren't pure ISA
    "#6502#cc65.corpus",  # Compiler-specific variant
]

import os
import sys
from pathlib import Path


def main():
    # Get the corpus directory path
    cpu_rec_dir = Path("./cpu_rec/cpu_rec_corpus")

    if not cpu_rec_dir.exists():
        print(f"Error: Directory {cpu_rec_dir} not found")
        sys.exit(1)

    # Print summary of actions
    print("Files to be renamed (to match other dataset):")
    for old_name, new_name in sorted(rename_map.items()):
        print(f"  {old_name} -> {new_name}")

    print("\nFiles to be removed (non-ISA specific):")
    for file in sorted(files_to_remove):
        print(f"  {file}")

    print(
        "\nAll other architecture corpus files will be kept with their original names."
    )

    # Confirm with user
    response = input("\nProceed with renaming and removal? (y/n): ")
    if response.lower() != "y":
        print("Operation cancelled.")
        return

    # Remove non-ISA files
    print("\nRemoving non-ISA files...")
    for file in files_to_remove:
        file_path = cpu_rec_dir / file
        if file_path.exists():
            file_path.unlink()
            print(f"Removed: {file}")
        else:
            print(f"Warning: {file} not found")

    # Rename files that map to the other dataset
    print("\nRenaming files...")
    for old_name, new_name in rename_map.items():
        old_path = cpu_rec_dir / old_name
        new_path = cpu_rec_dir / new_name
        if old_path.exists():
            old_path.rename(new_path)
            print(f"Renamed: {old_name} -> {new_name}")
        else:
            print(f"Warning: {old_name} not found")


if __name__ == "__main__":
    main()
