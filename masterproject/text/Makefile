PANDOC_FLAGS = \
	--resource-path=content \
	--citeproc \
	--csl="assets/ieee.csl" \
	--from=markdown+tex_math_single_backslash+tex_math_dollars+raw_tex \
	--to=latex \
	--standalone

.PHONY: wordcount watch worddiff pdf

pdf: ../../output/master-thesis.pdf

../../output/master-thesis.pdf: content/*.md 
	@start_time=$$(date +%s.%N); \
	echo "\nCompiling PDF... - $$(date +"%H:%M:%S")"; \
	mkdir -p ../../output; \
	pandoc content/*.md \
		$(PANDOC_FLAGS) \
		--pdf-engine=xelatex \
		--output=$@; \
	end_time=$$(date +%s.%N); \
	duration=$$(awk "BEGIN {printf \"%.2f\", $$end_time - $$start_time}"); \
	echo "Finished compiling (took $$duration seconds)"

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

latex: ../../output/master-thesis.tex

../../output/master-thesis.tex: content/*.md
	@echo "\nCompiling LaTeX file..."
	mkdir -p ../../output
	rm -f ../../output/master-thesis.tex
	pandoc content/*.md \
		$(PANDOC_FLAGS) \
		--output=$@
	@echo "Finished compiling LaTeX file."

wordcount:
	@cd content && find . -name "*.md" | grep -v -E "(00|9[0-9])-" | sort | xargs wc -w

worddiff:
	@./assets/wordcount-diff.sh

# make wordiff-3 for 3 commits back
worddiff-%:
	@./assets/wordcount-diff.sh $*
