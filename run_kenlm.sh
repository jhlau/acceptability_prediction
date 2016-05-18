#script to run kenlm
#checkout kenlm and compile the code (it needs boost library to build, check kenlm/BUILDING), e.g.
#git clone https://github.com/vchahun/kenlm
#cd kenlm; ./bjam; cd ..

#parameters
kenlm_dir="kenlm"
order=4
train_corpus="example_dataset/cleaned-normal/train_dir/train.txt"
test_corpus="example_dataset/cleaned-normal/test.txt"
#gold standard human ratings
gs="example_dataset/raw/test_gold_ratings.txt"
output_dir="output/kenlm"

#create output directory if necessary
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

echo 'Training N-gram language model...'
time $kenlm_dir/bin/lmplz -o $order -T $output_dir/tmp_lm -S 15G --vocab_file $output_dir/vocab --verbose_header \
    < $train_corpus > $output_dir/model.arpa

$kenlm_dir/bin/build_binary $output_dir/model.arpa $output_dir/model.klm

echo 'Computing the correlation of the computed scores and the gold standard human ratings...'
python calc_correlation_kenlm.py $test_corpus $gs $output_dir
