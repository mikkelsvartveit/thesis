UNAME := $(shell uname -s)

pdf:
	@echo "\nCompiling PDF..."
	mkdir -p ../output
	pandoc content/*.md \
		--resource-path=content \
		--citeproc \
		--template "assets/template.tex" \
		--csl="assets/ieee.csl" \
		--pdf-engine=xelatex \
		--from=markdown+tex_math_single_backslash+tex_math_dollars+raw_tex \
		--to=latex \
		--output=../output/preproject.pdf
	@echo "Finished compiling PDF."

watch:
	@make -s pdf
	@echo "Watching for changes..."
ifeq ($(shell uname -s),Linux)
	@inotifywait -r -m -e modify ./content/ | \
	while read file_path file_event file_name; do  \
		make -s pdf; \
	done
else
	@fswatch -0 ./content/ | xargs -0 -n 1 -I {} make -s pdf
endif

watchwsl:
	@make -s pdf
	@echo "Watching for changes..."
	inotifywait -r -m -e modify ./content/ | \
   while read file_path file_event file_name; do  \
      make -s pdf; \
   done 

latex:
	@echo "\nCompiling LaTeX file..."
	mkdir -p ../output
	pandoc content/*.md \
		--resource-path=content \
		--citeproc \
		--template "assets/template.tex" \
		--csl="assets/ieee.csl" \
		--from=markdown+tex_math_single_backslash+tex_math_dollars+raw_tex \
		--to=latex \
		--standalone \
		--output=../output/preproject.tex
	@echo "Finished compiling LaTeX file."

wordcount:
	@cd content && find . -name "*.md" | grep -v -E "(00|9[0-9])-" | sort | xargs wc -w