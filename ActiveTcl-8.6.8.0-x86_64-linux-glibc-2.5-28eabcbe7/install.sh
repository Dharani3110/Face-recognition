#!/bin/sh
# Shell wrapper around the main installation script.  Will use the
# wish or tclsh inside of the distribution to execute this
# installer.
#
# Auto-detects the location of the distribution. Auto-detects
# availability of X and chooses between gui and terminal based
# installers using this information.

dist=`dirname $0`

# initialize the environment so that the distributed "wish" and
# "tclsh" are able to find their libraries despite being crippled with
# the special path value we will replace later during the installation
# with the actual path to the installation.

TCL_LIBRARY=$dist/payload/lib/tcl8.6
TK_LIBRARY=$dist/payload/lib/tk8.6
LD_LIBRARY_PATH=$dist/payload/lib
DYLD_LIBRARY_PATH=$dist/payload/lib
SHLIB_PATH=$dist/payload/lib
LIBPATH=$dist/payload/lib

export TCL_LIBRARY TK_LIBRARY LD_LIBRARY_PATH DYLD_LIBRARY_PATH SHLIB_PATH LIBPATH

# Update_check
if [ -x $dist/update_check ]
then
  $dist/update_check > /dev/null 2>&1
  if [ 1 -eq $? ]
  then
    echo "Installer superceded"
    exit 1
  fi
fi

$dist/payload/bin/tclsh8.6 $dist/install.tcl "$@"

# pwd = inside the unpacked distribution ...
# go one level up and remove the directory
#cd ..
#rm -rf $dist

if [ -x $dist/komodo_download ]
then
  $dist/komodo_download
fi


exit
