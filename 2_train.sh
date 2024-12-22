loops=(0 1 2 3 4 5 6 7 8 9 10 11)
for loop in ${loops[@]}
do
    python src/6_retrieving_results.py
    python src/7_average.py
done