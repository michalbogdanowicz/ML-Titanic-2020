import csv


def read_file(fine_name):
    toReturn = [];
    with open(fine_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            toReturn.append(row)
    return toReturn

"""The objective is to join on ticket id  the survival rate into the clean test data."""
if __name__ == "__main__":
    # read the data from test_clean
    test_data = read_file('test_clean.csv')
    print(test_data)
    print('zjokan')
    # read the data from the whole dataset.
    all_data = read_file('titanic.csv')
    print(all_data)
    # join survival rate from whole dataset to the test
    # ticket can be used.

    # read from the clean

    # cocatenate train with test.

    # save as new clean_complete.


