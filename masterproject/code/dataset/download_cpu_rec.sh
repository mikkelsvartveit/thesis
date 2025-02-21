rm -rf cpu_rec
echo "Downloading CPU Rec dataset..."
wget -q --show-progress https://github.com/airbus-seclab/cpu_rec/archive/refs/heads/master.zip
echo "Unzipping..."
unzip -q master.zip
rm master.zip
mkdir cpu_rec
mv cpu_rec-master/cpu_rec_corpus cpu_rec/cpu_rec_corpus
rm -rf cpu_rec-master

cd cpu_rec/cpu_rec_corpus

echo "Decompressing files..."
for file in *.xz; do
    xz -d -q $file
done
echo "Done."

# rm files that starts with _
rm _*

cd ../../

python rename_cpu_rec.py