import streamlit as st
import pickle
import pandas as pd
import requests
import os

if not os.path.exists("similarity.pkl") or not os.path.exists("movie_dict.pkl"):
    import generate_model
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl', 'rb'))
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()

        data = response.json()

        print(data)  

        poster_path = data.get("poster_path")

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"

        return None

    except Exception as e:
        print("ERROR:", e)
        return None

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = []
    recommended_movies_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        print(movie_id)
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movie_id))
    return recommended_movies , recommended_movies_posters

st.title("Movie Recommendation System")


selected_movie_name = st.selectbox(
    'Choose a movie to recommend',
    movies['title'].values
)
if st.button('Recommend'):
    names , poster = recommend(selected_movie_name)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(names[0])
        if poster[0]:
            st.image(poster[0])

    with col2:
        st.text(names[1])
        if poster[1]:
            st.image(poster[1])

    with col3:
        st.text(names[2])
        if poster[2]:
            st.image(poster[2])

    with col4:
        st.text(names[3])
        if poster[3]:
            st.image(poster[3])

    with col5:
        st.text(names[4])
        if poster[4]:
            st.image(poster[4])
