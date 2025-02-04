# Background

TODO: Introduce section

## Computer software

### Binary executables

All computer software boils down to a series of bytes readable by the CPU. The bytes are organized in _instructions_. An instruction always includes an _opcode_ (Operation Code), which tells the CPU what operation should be executed. Depending on the opcode, the instruction often contains one or more _operands_, which provides the CPU with the data that should be operated on. The operands can be immediate values (values specified directly in the instruction), registers (a small, very fast memory located physically on the CPU), or memory addresses.

![Instruction format and examples from the ARM instruction set.](images/arm-instruction.svg)

<!-- Assembly? -->

### Instruction set architectures

\ac{ISA} yikes

\ac{ISA}
\ac{ISA}
\ac{ISA}

### Compilers

Software developers employ tools like compilers and interpreters to convert programs from human-readable programming languages to executable machine code. In the early days of computer programming, software had to be written in assembly languages that mapped instructions directly to binary code for execution. Growing hardware capabilities allowed for more complex software systems, however the lack of human readability of assembly languages made software increasingly difficult and expensive to maintain. In order to overcome this challenge, compilers were created to translate human-readable higher level languages like C into binary executables. Compilers are able to analyze the semantic meaning of the program, usually through some form of intermediate representation. The ISA of the target computer then provide the compiler with the recipe to translate the intermediate representation into executable code. The intermediate representation is usually language and system architecture agnostic, which has the added benefit of allowing a compiler to translate the same program to many different computer architectures.

<!--
Abstraction away from executables. without access to the original source code, it is dificult to understand waht a binary program does. Motivates reverse engineering.
 -->
