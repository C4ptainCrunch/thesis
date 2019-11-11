index.rst: index.ipynb
	jupyter-nbconvert --to rst index.ipynb --output=index.rst --template=celltemplate.tpl

watch_notebooks:
	@while true; do \
		make index.rst; \
		inotifywait -qre close_write .; \
	done
