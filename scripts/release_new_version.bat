cd C:\Users\Pedro\Desktop\nucml\
python setup.py sdist bdist_wheel
twine upload dist/* --skip-existing