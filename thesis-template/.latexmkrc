# pdf_mode hace que `latexmk -c` (make paper-clean) analice la regla de pdflatex y borre tambien
# los .aux de los \include; bibtex_use = 2 borra ademas el main.bbl.
$pdf_mode = 1;
$bibtex_use = 2;
