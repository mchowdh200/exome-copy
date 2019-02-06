set -exuo pipefail
export TMPDIR=/mnt/local
sudo apt-get update
sudo apt-get install -y build-essential git libfuse-dev libcurl4-openssl-dev libxml2-dev mime-support automake libtool pkg-config libssl-dev ncurses-dev awscli python-pip libbz2-dev liblzma-dev unzip openjdk-8-jre-headless
sudo mkfs -t ext4 /dev/nvme0n1
sudo mkdir /mnt/local
sudo mount /dev/nvme0n1 /mnt/local
sudo chown ubuntu /mnt/local
cd $TMPDIR
mkdir bin
cd bin
export PATH=$PATH:`pwd`
export BIN_DIR=`pwd`

wget https://github.com/brentp/gargs/releases/download/v0.3.8/gargs_linux
mv gargs_linux gargs
chmod +x gargs

wget http://home.chpc.utah.edu/~u6000771/excord
chmod +x excord

wget https://github.com/kahing/goofys/releases/download/v0.0.13/goofys
chmod +x goofys
cd $TMPDIR

git clone https://github.com/lh3/bwa.git
cd bwa; make
ln -s "$(pwd)/bwa" $BIN_DIR
cd $TMPDIR

git clone git://github.com/GregoryFaust/samblaster.git
cd samblaster
make
ln -s "$(pwd)/samblaster" $BIN_DIR
cd $TMPDIR

git clone https://github.com/samtools/htslib.git
cd htslib/
autoheader
autoconf
./configure  --disable-bz2  --disable-lzma
make
ln -s "$(pwd)/bgzip" $BIN_DIR
ln -s "$(pwd)/tabix" $BIN_DIR
cd $TMPDIR

wget https://github.com/samtools/samtools/releases/download/1.5/samtools-1.5.tar.bz2
bunzip2 samtools-1.5.tar.bz2
tar -xvf samtools-1.5.tar
cd samtools-1.5/
./configure  --disable-bz2  --disable-lzma
make
ln -s "$(pwd)/samtools" $BIN_DIR

wget https://github.com/samtools/bcftools/releases/download/1.5/bcftools-1.5.tar.bz2
bunzip2 bcftools-1.5.tar.bz2
tar -xvf bcftools-1.5.tar
cd bcftools-1.5/
./configure
make
ln -s "$(pwd)/bcftools" $BIN_DIR
cd $TMPDIR

# add mosdepth
mkdir mosdepth
cd mosdepth
wget https://github.com/brentp/mosdepth/releases/download/v0.2.4/mosdepth
ln -s "$(pwd)/mosdepth" $BIN_DIR
cd $TMPDIR


pip install svtools
pip install matplotlib



