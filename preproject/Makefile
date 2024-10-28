pdf:
	mkdir -p ../output
	@echo "Compiling PDF..."
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
	@fswatch -o ./content/ | xargs -n1 -I{} make -s pdf

latex:
	@echo "Compiling LaTeX file..."
	mkdir -p ../output
	pandoc content/*.md \
		--resource-path=content \
		--citeproc \
		--template "assets/template.tex" \
		--csl="assets/ieee.csl" \
		--from=markdown+tex_math_single_backslash+tex_math_dollars+raw_tex \
		--to=latex \
		--standalone \
		--output=../output/preproject.tex \
	@echo "Finished compiling LaTeX file."