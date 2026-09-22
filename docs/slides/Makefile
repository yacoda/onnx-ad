# XeLaTeX, twice: the diagrams are absolutely placed with TikZ `remember picture`,
# whose page coordinates are only resolved on the second pass. One pass silently
# puts every figure in the wrong place rather than failing.
slides.pdf: slides.tex figures.tex cstheme.sty
	xelatex -interaction=nonstopmode slides.tex
	xelatex -interaction=nonstopmode slides.tex

clean:
	rm -f slides.aux slides.log slides.nav slides.out slides.snm slides.toc

.PHONY: clean
