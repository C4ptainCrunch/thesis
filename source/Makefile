index.rst: index.ipynb
	jupyter-nbconvert --to rst index.ipynb --output=index.rst --template=celltemplate.tpl

lib/results.rst: lib/results.ipynb
	jupyter-nbconvert --to rst lib/results.ipynb --output=results.rst --template=celltemplate.tpl

lib/utils.rst: lib/utils.ipynb
	jupyter-nbconvert --to rst lib/utils.ipynb --output=utils.rst --template=celltemplate.tpl

watch_notebooks:
	@while true; do \
		make index.rst; \
		make lib/results.rst; \
		make lib/utils.rst; \
		inotifywait -qre close_write .; \
	done
