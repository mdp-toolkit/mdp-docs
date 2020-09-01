### Please refer to the online documentation at https://mdp-toolkit.github.io/.

### The source code is available at https://github.com/mdp-toolkit/mdp-toolkit.


### How to build the legacy documentation:

```bash
python2.7 -m virtualenv /PATH/TO/VIRTUALENV/
source /PATH/TO/VIRTUALENV/bin/activate
pip install sphinx==1.6.4 epydoc==3.0.1 numpy==1.16.6 future==0.18.2 scikit-learn==0.20.4 pp==1.6.5 joblib==0.14.1
# clone the repo and make sure the source code is cloned from the submodule
git clone --recursive https://github.com/mdp-toolkit/mdp-docs
# run the following in ./mdp-docs to build a local version of the docs
make legacywebsitelocal
# run the following in ./mdp-docs instead to build a version with weblinks
make legacywebsite
```
