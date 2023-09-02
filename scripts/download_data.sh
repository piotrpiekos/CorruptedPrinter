curl "https://dl.dropboxusercontent.com/scl/fi/jgmjqrzg8yy2jkqzifjxh/data_ref_1_1.csv?rlkey=ud8l5pwx2eh85xvekp2cv2o2y&dl=0" -o data/circles/data1.csv
curl "https://dl.dropboxusercontent.com/scl/fi/kv8t5cggx6tgb7p6imjf9/data_ref_1_2.csv?rlkey=eq2rvgfw08wxa4y8fhkvjyrh3&dl=0" -o data/circles/data2.csv
curl "https://dl.dropboxusercontent.com/scl/fi/60bmyxwujc62cx21adzio/data_ref_1_3.csv?rlkey=na2u0fn4qp6b3slonsr1bgcvd&dl=0" -o data/circles/data3.csv
curl "https://dl.dropboxusercontent.com/scl/fi/6qj7cikccwtox4u5y3ngp/data_ref_1_4.csv?rlkey=iorif6b2og96a8o7x8lm9at4a&dl=0" -o data/circles/data4.csv
curl "https://dl.dropboxusercontent.com/scl/fi/6sgp53l6yb3z0punl5q43/data_ref_1_5.csv?rlkey=4c96xgyqdig3jxera2o5mx92m&dl=0" -o data/circles/data5.csv
curl "https://dl.dropboxusercontent.com/scl/fi/ml1biuk93tprd4ry3fgv8/data_ref_1_6.csv?rlkey=um13e9krchykzqfyao42wvuw5&dl=0" -o data/circles/data6.csv
curl "https://dl.dropboxusercontent.com/scl/fi/i4bcsjpzg0wdyr30pnh46/data_ref_1_7.csv?rlkey=t9ganciup7wwpt6iz5tdbg3cb&dl=0" -o data/circles/data7.csv

cat data/circles/data*.csv >> data/circles/all_data.csv
rm data/circles/data*.csv


