import streamlit as st 
import pandas as pd
from io import StringIO, BytesIO 
from transformers import BertTokenizer, BertModel, AutoTokenizer
# from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Match. Mentor",
)

st.title('Match.Mentor')

tab_1, tab_2 = st.tabs(['Connect', 'Learn'])

######################################:
# Load pre-trained BERT model and tokenizer

# tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
# model = BertModel.from_pretrained ('google-bert/bert-base-uncased')

tokenizer = AutoTokenizer.from_pretrained('./bert-mini')
model = BertModel. from_pretrained('./bert-mini')


# Created dictionaries for every row
def create_sentences(df):
    # Initialize an empty list to store dictionaries
    dict_list = []
    # Iterate over each row of the DataFrame
    for index, row in df.iterrows():
        # Convert the row to a dictionary and append to the list
        row_dict = dict(row)
        dict_list.append(row_dict)
    return dict_list

# Encode input text and get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1) # Mean pooling over tokens
    return embeddings

# Calculate cosine similarity between two vectors
def cosine_similarity_vec(vec1, vec2) :
    return cosine_similarity(vec1.detach().numpy (), vec2.detach().numpy())

################################################################################
with st.sidebar:
    st.markdown ('#### Define weights for criteria')
    location_weight = st.slider (
        "Location",
        min_value = 0.0,
        max_value = 1.0,
        step = 0.1,
        value = 0.6
    )

    # Define weights for criteria
    weights = {'Location': location_weight,}

    st.markdown('#### Number of recommendations to provide') 
    number_of_recommendations = st.number_input('Insert a number', value=3)

    st. markdown("""
        #### How-To
        1.    Define weights for criteria
        2.    Set the number of recommendations
        3.    Upload files containing mentors and mentees
        4.    Select a mentor
        5.    Review recommendations
                
        ### :red[Important]  
        Files should be in the following format:  
        | Name | Summary | Location |
        |-|-|-|
        |First Last|Insert Summary| Baltimore, MD|
    """)

with tab_1:
    # read files
    st.markdown("**Upload a file containing the list of mentors**")
    mentors_file = st.file_uploader('Choose a mentor file',
                                    label_visibility='collapsed')

    st.markdown("**Upload a file containing the list of mentees**")
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

        st.markdown(f"""
            ---  
            **Results**  
            **There are {mentors_df.shape[0]} mentors and \
            {mentees_df.shape[0]} mentees.**
        """)

        # generate sentences for mentors
        mentor_dataset = create_sentences(mentors_df)

       # generate sentences for mentees
        mentee_dataset = create_sentences(mentees_df)

        st.markdown(f"Select a mentor to see possible matches")
        mentor_to_use = st.selectbox(label='Mentors',
                                     options=mentor_dataset,
                                     format_func=lambda mentor: mentor.get('Name'))
        
        key = 'Name'

        index = next((index for index, d in enumerate(mentor_dataset) if d.get(str(key)) == mentor_to_use.get(str(key))), None)

        # get mentor using index
        mentor = mentor_dataset[index]

        input_embedding = get_bert_embedding(mentor['Summary'])

        adjusted_similarities = {}
        mylist = []
        for item in mentee_dataset:
            text = item['Summary'] # extract text from dictionary
            other_embedding = get_bert_embedding(text)
            similarity_score = cosine_similarity_vec(input_embedding, other_embedding)

            # apply adjustment based on criteria
            adjusted_score = similarity_score
            for criterion, weight in weights.items():
                if criterion in mentor and mentor[criterion] == item.get(criterion):
                    adjusted_score += weight * .1 # Adjust on a factor based on criterion importance
                    # the intention is to adjust the similarity score based on the importance of the criterion
                    # if a certain criterion matches the input criteria, we want to boost the similarity score
                    # by a certain percentage of the criterion's weight. 
                    # 'weight' represents the importance of the criterion, and .1 is just an arbitrary scaling factor
                    # to adjust the impact of the criterion on the similarity score. (.1 is 10% of the weight)
                    # The purpose of the scaling factor is to ensure that the adjustments based on criteria 
                    # are meaningful relative to the original similarity scores. Multiplying by a small fraction
                    # like .1 ensures that the adjustments don't dominate the original similarity scores.
                    # If the adjustment based on criteria are too subtle or dominant, the scaling factor
                    # can be increased or decreased accordingly. Different scaling factors can help fine-tune
                    # the recommendation model
                    item['Criteria Match'] = 'Summary, ' + criterion
                else: 
                    item['Criteria Match'] = 'Summary only'
            
            # adjusted similarities[text] = adjusted_score # Store similarity scores by text
            adjusted_similarities[item['Name']] = adjusted_score

            item['Similarity Score'] = adjusted_score 
            mylist.append(item)

        st.markdown(f"**Top {number_of_recommendations} matches for {mentor_to_use.get('Name')}**")
                    
        # Get top N recommendations based on adjusted similarity scores (x[1] second element in each tuple i.e. similarity score)
        # top_recommendations = sorted(adjusted_similarities.items(), key-lambda x: x[1], reverse-True) [:number _of_recommendations]
        # st.write(f"Top {number_of_recommendations} recommendations:", top recommendations)

        # Get top N recommendations based on adjusted similarity scores (×['Similarity Score'])
        sorted_results = sorted(mylist, key = lambda x: x['Similarity Score'], reverse=True)

        for index, mentee in enumerate(sorted_results[:number_of_recommendations]):
            if 'only' not in mentee['Criteria Match']:
                st.markdown(f""":green[
                    **{index+1}**
                    Name: {mentee['Name']}
                    Criteria Match: {mentee['Criteria Match']}
                    Similarity Score: {mentee['Similarity Score'][0][0]}]
                """)
            else:
                st.markdown (f"""
                    **{index+1}**
                    Name: {mentee['Name']}
                    Criteria Match: {mentee['Criteria Match']}
                    Similarity Score: {mentee['Similarity Score'][0][0]}
                """)

with tab_2:
    st.markdown("""
    **Match.Mentor**  
                
    **Description**  
    Mentor Match is an application that leverages BERT (Bidirectional Encoder
    Representations from Transformers) model to facilitate seamless mentor-mentee
    connections. The app streamlines the process of identifying potential mentees, 
    enabling mentors to discover ideal matches effortlessly based on their unique 
    preferences and criteria.
                
    **Benefits**  
    - Empowers mentors to identify promising mentees efficiently and effectively.
    - Facilitates meaningful connections that foster professional growth and development.
    - Saves time and effort by automating the process of mentor-mentee pairing.
    - Provides tailored and relevant match suggestions.
                
    **Similarity Score and Criteria Weights**  
    Similarly score is the cosine similarity between the two vectors (one for a mentor and one for a mentee).
    The higher the similarity score, the more similar the two vectors (or text inputs) are.
    In the context of this recommendation system, higher similarity scores imply that the recommended items (mentees) 
    are more like the input item (mentor), making them better recommendations in terms of similarity.  
                
    Weight represents the importance of the criterion. I use an arbitrary scaling factor of .1 
    to adjust the impact of the criterion on the similarity score (1 or 10% of the weight).
    The intention is to adjust the similarity score based on the importance of the criterion.
    If a certain criterion matches the input criteria, then I boost the similarity score by a
    certain percentage of the criterion's weight.
    The purpose of the scaling factor is to ensure the adjustments based on criteria are meaningful 
    relative to the original similarity scores. Multiplying by a small fraction like .1 ensures that 
    the adjustments don't dominate the original similarity scores.
    """)