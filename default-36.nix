with import <nixpkgs> {};

stdenv.mkDerivation rec {
  name = "env36";
  env = buildEnv { name = name; paths = buildInputs; };
  buildInputs = [
    python3
    python36Packages.tensorflowWithoutCuda
    python36Packages.ipython
    python36Packages.nose
    python36Packages.urllib3
    python36Packages.pillow
    python36Packages.Keras
    python36Packages.h5py
    python36Packages.pandas
    #python36Packages.pyspark
    python36Packages.parameterized
    jdk
    which
    # For the notebooks:
    python36Packages.notebook
    python36Packages.matplotlib
  ];
}