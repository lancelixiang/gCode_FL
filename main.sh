types=('data' 'StratifiedKFoldData')
for type in ${types[@]}
do
    idxs=(0 1 2 3)
    for idx in ${idxs[@]}
    do
        python src/1_create_assets.py --idx $idx --type $type
        python src/3_create_data_users.py
        python src/4_init_code.py
        python src/5_review_code_request.py

        loops=(0 1 2 3 4 5 6 7 8 9 10 11 0 1 2 3 4 5 6 7 8 9 10 11 0 1 2 3 4 5 6 7 8 9 10 11 0 1 2 3 4 5 6 7 8 9 10 11)
        for loop in ${loops[@]}
        do
            python src/6_retrieving_results.py
            python src/7_average.py  --idx $idx  --type $type
        done
    done
done


