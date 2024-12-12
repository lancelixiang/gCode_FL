loops=(0 1 2 3 4 5)
for loop in ${loops[@]}
do
    python src/6_retrieving_results.py
    python src/7_average.py
done