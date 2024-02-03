all:
	python3 setup.py install>out.txt 

step_test:
	python3 test/step2.py
	
convert_test:
	python3 test/convert.py
# clean the temporary files, rebuild the module and 
# reinstall the new module.
reset:
	python3 setup.py clean --all
	python3 setup.py build
	pip install .

install:
	python3 setup.py install

user_install:
	python3 setup.py install --user

clean:
	rm -r out.txt *.csv 