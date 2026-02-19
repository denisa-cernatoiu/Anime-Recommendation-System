from flask import Flask, render_template, g, request, flash
from flask_bootstrap import Bootstrap5
from recommender import AnimeRecommendation
import sqlite3
import os

app = Flask(__name__)
Bootstrap5(app)
app.secret_key = os.environ.get("SECRET_KEY", "dev-key")

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


# making recommendations from the db based on user input
@app.route('/search-suggestions')
def search():
    query = request.args.get("search", "")
    db = get_db()
    cursor = db.cursor()

    # case insensitive search
    cursor.execute("""
        SELECT title_english, coverImage_large
        FROM anime
        WHERE title_english LIKE ? COLLATE NOCASE
        LIMIT 5
    """, (f"%{query}%",)) 

    suggestions = cursor.fetchall()

    db.close()
    
    return render_template("search_results.html", suggestions=suggestions)


# getting the name of the selected anime and filling the input form with it
@app.route("/fill-input")
def fill_input():
    selected = request.args.get("selected", "")
    return  f'''<input 
        type="text" 
        name="search" 
        id="search-input"
        value="{selected}"
        placeholder="Enter an anime name" 
        autocomplete="off" 
        hx-get="/search-suggestions" 
        hx-trigger="keyup changed delay:500ms" 
        hx-target="#suggestions-list"
        hx-swap="innerHTML"/>'''


# showing the user his recommendations
@app.route("/anime_list")
def anime_list():
    db = get_db()
    cursor = db.cursor()

    anime_suggested = None
    
    selected_anime = request.args.get("search", "")

    cursor.execute(f"""
        SELECT title_english
        FROM anime
        WHERE title_english = ?
    """, (selected_anime,))

    is_in_db = cursor.fetchone()

    # if the anime is in the db, get recommendations, otherwise show an error page
    if is_in_db:
        anime_suggested = rec.get_recommendations(selected_anime)
    
    else:
        return render_template("error_page.html")


    placeholders = ",".join("?" * len(anime_suggested))

    cursor.execute(f"""
        SELECT title_english, rating, description, coverImage_large 
        FROM anime
        WHERE title_english IN ({placeholders})
    """, anime_suggested)

    animes = cursor.fetchall()
    db.close()


    return render_template("second_page.html", animes=animes)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

