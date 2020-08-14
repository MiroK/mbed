# Mbed

## Dependencies
You will need `networkx`, `tqdm`, `perfusion` in addition to usual scientific
Python stack. For the latter see [GitHub](https://github.com/MiroK/perfusion). 
The most crucial dependency is Gmsh. You should get a development (SDK) version from 
[here](http://gmsh.info/bin/Linux/). Currently supported one is [version](http://gmsh.info/bin/Linux/gmsh-4.4.1-Linux64-sdk.tgz). Once untarred the gmsh (lib) folder with `gmsh.py` needs to be put on python path.
For example

```bash
export PYTHONPATH="/home/mirok/Documents/Software/gmsh-4.4.1-Linux64-sdk/lib":"$PYTHONPATH"
```

## Installation
Finally, put the `mbed` module on python path. For the current shell session it suffices to `source setup.rc`



