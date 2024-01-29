all:
	python3 setup.py install>out.txt 

test:
	python3 nonl.py

# clean the temporary files, rebuild the module and 
# reinstall the new module.
reset:
	python3 setup.py clean --all
	python3 setup.py build
	pip install .

clean:
	rm -r out.txt *.csv 