#/bin/bash

if [ -d "UTMOSv2" ]; then
    rm -rf UTMOSv2
fi

# Check if git-lfs is installed
if git lfs --version >/dev/null 2>&1; then
  echo "Git LFS is installed."
  git-lfs install
else
  echo "Git LFS is not installed. Please install git-lfs first."
  echo "You may check tools/install_gitlfs.md for guidelines"
  exit 1
fi

GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/ftshijt/UTMOSv2.git
cd UTMOSv2
# Prevents LFS files from being temporarily downloaded during the installation process
GIT_LFS_SKIP_SMUDGE=1 pip install -e .
cd ..

echo "Be aware to execute `source activate_utmosv2.sh` to enable the checkpoint"
