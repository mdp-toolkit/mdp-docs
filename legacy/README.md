How to built the legacy documentation
=====================================

```bash
python2.7 -m virtualenv /PATH/TO/VIRTUALENV/
source /PATH/TO/VIRTUALENV/bin/activate
pip install sphinx==1.6.4 epydoc==3.0.1 numpy==1.16.6 future==0.18.2 scikit-learn==0.20.4 pp==1.6.5 joblib==0.14.1
# run the following in ./legacy to build a local version of the docs
make legacywebsitelocal
# run the following in ./legacy instead to build a version with weblinks
make legacywebsite
```
