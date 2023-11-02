import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from data.helpers import create_sentences, join_mentor_to_mentees,\
                         get_name_dict, get_index
from models.predict_model import bert

st.title('Match.Metor')

app_tab, info_tab = st.tabs(['Home', 'Information'])

with app_tab:
    # read files
    st.markdown("**Upload a file containg the list of mentors**")
    mentors_file = st.file_uploader('Choose a mentor file',
                                    label_visibility='collapsed')

    st.markdown("**Upload a file containg the list of mentees**")
    mentees_file = st.file_uploader('Choose a mentee file', 
                                    label_visibility='collapsed')
    
    # summary of the files
    if mentors_file is not None and mentees_file is not None:
        # Read mentors file
        # To read file as bytes:
        bytes_data = mentors_file.getvalue()
        # To convert to a string based IO:
        stringio = StringIO(mentors_file.getvalue().decode("utf-8"))
        # To read file as string:
        string_data = stringio.read()
        mentors_df = pd.read_csv(mentors_file)

        # Read mentees file
        # To read file as bytes:
        bytes_data = mentees_file.getvalue()
        # To convert to a string based IO:
        stringio = StringIO(mentees_file.getvalue().decode("utf-8"))
        # To read file as string:
        string_data = stringio.read()
        mentees_df = pd.read_csv(mentees_file)

        st.markdown(f"**There are {mentors_df.shape[0]} mentors and \
                    {mentees_df.shape[0]} mentees.**")

        st.markdown(f"Pick a mentor to see possible matches")
        mentor = st.selectbox('Mentor',
                           mentors_df['Name'])
        st.markdown(f"Showing matches for **{mentor}**")

        # generate sentence for mentors
        sentences_mentors = create_sentences(mentors_df)
        # generate sentences for mentees
        sentences_mentees = create_sentences(mentees_df)
        # add mentor to mentees
        sentences = join_mentor_to_mentees(sentences_mentors, sentences_mentees)
    
        ################################################################################
        # get dictionaty containing mentor and corresponding indices
        flipped_dict = get_name_dict(mentors_df)# what is this for?
        # get index for selected mentor
        index = get_index(mentor, flipped_dict)
        ################################################################################
        
        # find similarities using bert
        similarities = bert(sentences[index])
        # st.write(similarities) # write results 

        # convert list to dict
        my_dict = {index: value for index, value in enumerate(similarities)}

        #### TCHEKC THOS
        # Create a dictionary with keys in descending order by value
        sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1], reverse=True))
        #st.write(sorted_dict) # !!!this doesn't print the dictionaty in the correct order
        print(sorted_dict)

        ## get list of key indices
        mentee_keys_ordered = sorted_dict.keys()
        # Increasing all keys by 1
        increased_list = [k + 1 for k in mentee_keys_ordered]
        print(increased_list)

        ordered_list_mentees = [sentences[index][i] for i in increased_list]
        
        st.write('Matches in descending order')
        for mentee, confidence in zip(ordered_list_mentees, increased_list):
            st.write(f"{mentee} -- [Confidence: {sorted_dict[confidence-1]}]")
        #st.write(ordered_list_mentees) # write all sentences (mentor and mentees)
        # here


with info_tab:
    pass
