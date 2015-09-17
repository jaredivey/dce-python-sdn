# dce-python-sdn
This repository supplies the patches and configuration file to build the DCE-enabled SDN simulation framework in ns-3. Using bake to download and build the necessary libraries for this framework, it can set up the following libraries:

1. **ns-3-sdn** (github.com/jaredivey/ns-3-sdn): This version of ns-3 has added two modules to the ns-3 baseline: layer2-p2p and sdn. layer2-p2p provides a point-to-point channel and net device framework that maintains Ethernet headers instead of PPP headers. The sdn module holds all of the classes necessary to set up an SDN topology with/without DCE.
1. **ns-3-dce** (github.com/jaredivey/ns-3-dce): This version of DCE adds some minor updates based on glibc symbols not present in the DCE baseline. It also adds some examples in myscripts/sdn that demonstrate the use of the ns-3-sdn SdnSwitch with the Python-based controllers POX and Ryu.
1. **Python-2.7.10** (github.com/jaredivey/Python-2.7.10): The dependencies for the POX and Ryu libraries are built directly into this local Python library in order to reduce the possibility of path/dependency issues when running DCE. Additionally, the created Python library will be directly placed in files-0 filespace in ns-3-dce in order to be recognizable by a DCE-enabled node in ns-3-dce.
1. **POX** (github.com/noxrepo/pox): POX will be patched and directly placed in files-0 filespace in ns-3-dce in order to be recognizable by a DCE-enabled node in ns-3-dce.
1. **Ryu** (github.com/osrg/ryu): Ryu will be patched and directly placed in files-0 filespace in ns-3-dce in order to be recognizable by a DCE-enabled node in ns-3-dce.

NOTE: If more than one controller node is to be used, simply make copies of the files-0 directory in ns-3-dce, i.e. files-1, files-2, etc.

## Installation:

It is recommended that a VM be used for testing and experimenting, particularly due to the fact that the patched version of libfluid will be installed to the user's /usr directory. An appropriate choice of VM would be the one provided through SDNHub at http://sdnhub.org/tutorials/sdn-tutorial-vm/

1. Install ns-3 prerequisites:
  https://www.nsnam.org/wiki/Installation

1. Install libfluid prerequisites (debian package names provided):
  ```
  autoconf libtool build-essential pkg-config libevent-dev libssl-dev
  ```

1. Get bake:
  ```
  hg clone http://code.nsnam.org/bake bake
  export BAKE_HOME=`pwd`/bake
  export PATH=$PATH:$BAKE_HOME
  export PYTHONPATH=$PYTHONPATH:$BAKE_HOME
  ```

1. Get dce-python-sdn (Add -vvv to the last two instructions if problems occur during installation):
  ```
  git clone https://github.com/jaredivey/dce-python-sdn
  cd dce-python-sdn
  # Bake libfluid with elevated privileges
  bake.py configure -c bakeconf-sdn.xml -e libfluid
  bake.py download
  sudo ../bake/bake.py build
  # Bake dce-python-sdn
  bake.py configure -c bakeconf-sdn.xml -e dce-python-sdn
  bake.py download
  bake.py build
  ```

1. Go to the ns-3-dce directory and run an example script to test:
  ```
  cd source/ns-3-dce
  ./waf --run "dce-python-sdn --numSwitches=2"
  ```
