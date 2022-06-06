dataset_operation() {
    tail -n +2 imdb_top_1000.csv | shuf > imdb_top_1000.csv.s
    head -n $CUTOFF imdb_top_1000.csv.s > ./imdb_top_1000.csv.shuf
    len1=$(cat ./imdb_top_1000.csv.shuf | wc -l)
    len2=$(($len1/10))
    len3=$(($len2*2))
    len4=$(($len3+1))
    head -n $len2 imdb_top_1000.csv.shuf > imdb_top_1000_test.csv
    head -n $len3 imdb_top_1000.csv.shuf | tail -n $len2 > imdb_top_1000_dev.csv
    tail -n +$len4 imdb_top_1000.csv.shuf > imdb_top_1000_train.csv
    rm imdb_top_1000.csv.shuf
    wc -l imdb_top_1000.csv.*
}

kaggle datasets download -d harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows
unzip -o imdb-dataset-of-top-1000-movies-and-tv-shows.zip
dataset_operation