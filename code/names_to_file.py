from description_data import train_labels

def save_strings_to_file(strings, filename):
    """
    This function saves a list of strings to a file with each string on a new line.

    :param strings: list of strings
    :param filename: name of the file
    """
    with open(filename, 'w') as f:
        for item in strings:
            f.write("%s\n" % item)


# usage
filename = "companies.txt"
save_strings_to_file(train_labels, filename)
