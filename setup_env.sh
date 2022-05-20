mkdir external

#download self talk data splits
git clone https://github.com/vered1986/self_talk.git ./external/self_talk
cd external/self_talk
git lfs pull
cd -

#Download COPA
mkdir ./external/COPA
wget https://people.ict.usc.edu/~gordon/downloads/COPA-resources.tgz -O ./external/COPA/COPA-resources.tgz
cd external/COPA
tar xf COPA-resources.tgz
cd -

#Download SocialIQA
mkdir ./external/socialiqa
wget https://maartensap.github.io/social-iqa/data/socialIQa_v1.4.tgz -O ./external/socialiqa/socialIQa_v1.4.tgz
cd ./external/socialiqa
tar xf socialIQa_v1.4.tgz
cd -

#Download PIQA
mkdir ./external/piqa
cd ./external/piqa
wget https://yonatanbisk.com/piqa/data/train.jsonl
wget https://yonatanbisk.com/piqa/data/train-labels.lst
cd -

#Download WinoGrande
mkdir ./external/winogrande
cd ./external/winogrande
wget https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip
unzip winogrande_1.1.zip
cd -

#Download ARC
mkdir ./external/ARC
cd ./external/ARC
wget https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip
unzip ARC-V1-Feb2018.zip
cd -

#nltk
python -m nltk.downloader all

mkdir cached_output
mkdir results
