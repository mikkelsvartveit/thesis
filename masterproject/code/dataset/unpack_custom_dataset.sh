DIR=$(pwd)
if [[ "$DIR" != */dataset ]]; then
    echo "Error, please cd to the dataset directory"
    exit 1
fi

echo "Copying files from crosscompiler/results to buildcross"

rm -rf ./buildcross
mkdir -p ./buildcross
cp ../../crosscompiler/results/text_bin.tar.gz ./buildcross
cp ../../crosscompiler/results/labels.csv ./buildcross
cp ../../crosscompiler/results/report.txt ./buildcross


echo "Unpacking text_bin.tar.gz"

cd ./buildcross
tar -xzf text_bin.tar.gz
rm text_bin.tar.gz
cd ..

echo "Done"