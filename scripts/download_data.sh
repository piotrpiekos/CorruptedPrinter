curl "https://dl.dropboxusercontent.com/scl/fi/q1g92kxia56e29xmw9f0h/data_1_1.csv?dl=1&rlkey=b4vifcsy1wg34aiw8c4c2a44s" -o data/circles/data1.csv
curl "https://dl.dropboxusercontent.com/s/2myn2ksjcru04gu/data_1_2.csv?dl=0" -o data/circles/data2.csv
curl "https://dl.dropboxusercontent.com/s/391yrb3jo5t4ekx/data_1_3.csv?dl=0" -o data/circles/data3.csv
curl "https://dl.dropboxusercontent.com/scl/fi/ng6xxojikdsntq5zbjb9l/data_1_4.csv?rlkey=ssz2ex9j4frdvmxrhvwz0qtfa&dl=0" -o data/circles/data4.csv
curl "https://dl.dropboxusercontent.com/scl/fi/ftrsqxdsg306y91fifboi/data_1_5.csv?rlkey=e8de4rzax33x1wzfuxhfshggo&dl=0" -o data/circles/data5.csv
curl "https://dl.dropboxusercontent.com/scl/fi/z8sdyfuneyjs9esny9450/data_1_6.csv?rlkey=1xraev2u5tq4iek0iyml0k60a&dl=0" -o data/circles/data6.csv
curl "https://dl.dropboxusercontent.com/scl/fi/9q7tofqas3ibce5ysqzpk/data_1_7.csv?rlkey=zxqogpotx7jkw43enfl2snud9&dl=0" -o data/circles/data7.csv

cat data/circles/data*.csv >> data/circles/all_data.csv
rm data/circles/data*.csv
