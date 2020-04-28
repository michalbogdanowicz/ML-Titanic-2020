import csv


def read_file(fine_name):
    toReturn = [];
    with open(fine_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            toReturn.append(row)
    return toReturn

"""The objective is to join on passager id the survival rate."""
if __name__ == "main":
    # read the data from test_clean
    test_data = read_file('test_clean.csv')
    # read the data from the whole dataset.
    test_data = read_file('test_clean.csv')
    all_data = open("titanic.csv", "r")
    all_data_rows = all_data.readlines()
    # join survival rate from whole dataset to the test

    # read from the clean
    with open('train_clean.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row['first_name'], row['last_name'])
    # cocatenate train with test.

    # save as new clean_complete.


