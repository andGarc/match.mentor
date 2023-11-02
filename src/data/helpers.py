import pandas as pd

'''
Function to join dataframe row cell data into one sentence
Params:
    - df (dataframe): dataframe containging name, summary, and location info
Returns:
    - sentences (list): list of sentences for each row in the dataframe
'''
def create_sentences(df):
    # First let's join all 3 columns per row into one
    # Join the columns with a '-'
    df['joined_columns'] = df.apply(lambda x: ' - '.join(map(str, x[['Name', 'Summary', 'Location']])), axis=1)
    # Convert joined_columns to a list
    sentences = df['joined_columns'].to_list()
    return sentences

'''
Function to join a mentor to the mentees
Join one mentor sentence to all the sentences for the mentees
Params:
    - mentor (str) - sentence describing the mentor
    - mentee (list) - sentence describing all mentees
Returns:
    - sentences (list): joined mentor sentence to mentees sentences
                        (mentor plus mentees, mentors is always at [0]) 
**To-Do: optimize
'''
def join_mentor_to_mentees(mentors, mentees):
    sentences = []
    for mentor in mentors:
        sentences.append([mentor] + mentees)
    return sentences

'''
Function to get the key value pairs for a list of mentors or mentees
A key value pair is a dict where the key is the name and the value is the index
A dictionary is created from the Name column. Keys and values are then flipped
to get the desired structure - {'Name',  Index}
Params:
    - df (dataframe) - dataframe containging name, summary, and location info
Returns:
    - result (dict): key value pairs for mentor or mentees. 
                     Names are keys. Values are indices.
**To-Do: optimize
'''
def get_name_dict(df):
    name_dict = df['Name'].to_dict()
    # flip keys and values 
    flipped_name_dict = {value:key for key, value in name_dict.items()}
    return flipped_name_dict

'''
Funcition to get the index for a name using the flipped dictionary
Params:
    - name (str): name of a mentor/mentee to get the index for
    - flipped_dict (dict): dictionary containing the key value pairs
Return:
    - index (int): index of the name
'''
def get_index(name, flipped_dict):
    return flipped_dict[name]