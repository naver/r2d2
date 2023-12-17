'''
This python script writes a csv file with the list of images that are yet to be processed.
'''
import os
import re

data_dir= './data/oxbuild_images-v1/'
ls_data= list(os.listdir(data_dir))
print(len(ls_data))

# Compare the list of files in results directory
ls_incomplete= list(os.listdir('./results/oxbuild_images-v1_svd_n3t20/'))
# Define the regex pattern
pattern = re.compile(r'_SVDtau20')
# Iterate through each element in the list and remove the substring
for i in range(len(ls_incomplete)):
    ls_incomplete[i] = pattern.sub('', ls_incomplete[i])

# Print the modified list
print(len(ls_incomplete))

# How to subtract two lists?
to_do= list(set(ls_data) - set(ls_incomplete))
print(len(to_do))
# How to convert list to csv?
import csv
with open('./results/to_do.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(to_do)
