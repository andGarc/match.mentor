import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
from data.helpers import create_sentences, join_mentor_to_mentees,\
                         get_name_dict, get_index
from models.predict_model import bert

st.title('Match.Metor')

tab_1, tab_2, tab_3 = st.tabs(['Connect', 'Explore','Learn'])

with tab_1:
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
        st.markdown(f"Showing matches for **{mentor}** *(Best - Least)*")

        # generate sentence for mentors
        sentences_mentors = create_sentences(mentors_df)
        # generate sentences for mentees
        sentences_mentees = create_sentences(mentees_df)
        # add mentor to mentees
        sentences = join_mentor_to_mentees(sentences_mentors, sentences_mentees)
    
        # get dictionary with mentor and index
        # the mentor's name is the key and the value the index 
        # index is the order in the mentor file
        flipped_dict = get_name_dict(mentors_df)

        # get index for selected mentor from the dropdown
        index = get_index(mentor, flipped_dict)
        
        # find similarities using bert
        similarities = bert(sentences[index])

        # convert similarities list to dict
        my_dict = {index: value for index, value in enumerate(similarities)}

        #### Refactor this 
        # Create a dictionary with keys in descending order by value
        sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1], 
                                  reverse=True))

        ## get list of key indices
        mentee_keys_ordered = sorted_dict.keys()
        # Increasing all keys by 1
        increased_list = [k + 1 for k in mentee_keys_ordered]

        ordered_list_mentees = [sentences[index][i] for i in increased_list]
        
        for mentee, confidence in zip(ordered_list_mentees, increased_list):
            st.write(f"{mentee} -- [Confidence: {sorted_dict[confidence-1]}]")

with tab_2:
    st.markdown('**Mentor-Mentee Confidence Heatmap**')

    def create_heatmap(mentors, mentees, confidence_values):
        df = pd.DataFrame(confidence_values, index=mentors, columns=mentees)

        # Create a heatmap using seaborn with white to green color map
        plt.figure(figsize=(12, 8))
        heatmap = sns.heatmap(df, annot=True, cmap='YlGn', fmt=".2f", cbar_kws={'label': 'Confidence (%)'})

        # Set axis labels and plot title
        plt.xlabel('Mentees')
        plt.ylabel('Mentors')

        # Move x-axis label to the top
        heatmap.xaxis.set_label_position('top')
        heatmap.xaxis.tick_top()

        plt.title('Mentor-Mentee Confidence Heatmap')

        # Save the plot to a BytesIO object
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        plt.close()

        return img_bytes

    # prep dataframe for heatmap *** refactor
    # get list of mentor and mentee names
    list_of_mentor_names = mentors_df['Name'].tolist()
    list_of_mentee_names = mentees_df['Name'].tolist()

    confidence_values = [] # list to store list of confidence values

    for index, item in enumerate(list_of_mentor_names):
        similarities = bert(sentences[index])
        confidence_values.append(similarities)
   
    st.image(create_heatmap(list_of_mentor_names, list_of_mentee_names, 
                            confidence_values), 
                            caption='Heatmap', use_column_width=True)

with tab_3:
    st.markdown("""
**Match.Mentor**

**Description**

Match.Mentor is an application that harnesses the BERT (Bidirectional Encoder 
Representations from Transformers) model, to facilitate seamless mentor-mentee 
connections. The platform streamlines the process of identifying potential 
mentees, enabling mentors to discover ideal matches effortlessly based on their 
unique preferences and criteria.

**Key Features**

1. **BERT-Powered Matching Algorithm:** Utilizes BERT's sophisticated natural 
language processing capabilities to analyze mentor and mentee profiles, 
delivering highly accurate and relevant match suggestions.

**Benefits**

- Empowers mentors to identify promising mentees efficiently and effectively.
- Facilitates meaningful connections that foster professional growth and 
development.
- Saves time and effort by automating the process of mentor-mentee pairing.
- Enhances the overall mentorship experience through tailored and relevant 
match suggestions.

---
MIT License (2023)

""")
