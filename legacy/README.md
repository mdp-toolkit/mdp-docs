How to built the legacy documentation
=====================================

```bash
python2.7 -m virtualenv /PATH/TO/VIRTUALENV/
source /PATH/TO/VIRTUALENV/bin/activate
pip install sphinx==1.6.1 epydoc==3.0.1 numpy==1.16.6 future==0.18.2
# run the following in ./legacy
make codesnippet
make html
```
