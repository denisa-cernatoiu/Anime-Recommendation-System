from flask import Flask, render_template, g, request
from flask_bootstrap import Bootstrap5
from recommender import AnimeRecommendation
import sqlite3

app = Flask(__name__)
Bootstrap5(app)

# loading just once when the app starts
rec = AnimeRecommendation.load('model.pkl') 


def get_db():
    if '_database' not in g:
        g._database = sqlite3.connect("db.sqlite")
        g._database.row_factory = sqlite3.Row
    return g._database


@app.route("/")
def home():
    return render_template("first_page.html")


@app.route("/anime_list")
def anime_list():
    anime_suggested = None

    selected_anime = request.args.get("search", "")

    if selected_anime:
        anime_suggested = rec.get_recommendations(selected_anime)
    print(anime_suggested)

    db = get_db()
    cursor = db.cursor()

    placeholders = ",".join("?" * len(anime_suggested))

    cursor.execute(f"""
        SELECT title_english, rating, description, coverImage_large AS img_url
        FROM anime
        WHERE title_english IN ({placeholders})
    """, anime_suggested)

    movies = cursor.fetchall()
    db.close()


    return render_template("second_page.html", movies=movies)



if __name__ == "__main__":
    app.run(debug=True)

