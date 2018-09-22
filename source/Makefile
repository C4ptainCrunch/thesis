NOTEBOOKS := $(wildcard notebooks/*.ipynb)
RSTS := $(patsubst notebooks/%.ipynb,nb_builds/%.rst,$(NOTEBOOKS))

notebooks: $(RSTS)

nb_builds/%.rst: notebooks/%.ipynb
	jupyter-nbconvert --to rst '$<' --output='../$@' --template=celltemplate.tpl

watch_notebooks:
	@while true; do \
		make notebooks; \
		inotifywait -qre close_write .; \
	done
