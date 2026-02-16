from flask import Flask, render_template, g,  redirect, url_for, request
from flask_bootstrap import Bootstrap5
from reccomander import recommendations
import sqlite3
import pickle

app = Flask(__name__)
Bootstrap5(app)

cos_sim = pickle.load(open('cosine.pkl', 'rb'))


def get_db():
    if '_database' not in g:
        g._database = sqlite3.connect("db.sqlite")
    return g._database


def get_suggestions(client_input):
    db = get_db()
    cursor = db.cursor()

    cursor.execute("""
        SELECT title_english
        FROM anime
        WHERE title_english LIKE ?
        LIMIT 5
    """, (f"%{client_input}%",))

    return [row[0] for row in cursor.fetchall()] 


@app.route("/")
def home():
    return render_template("first_page.html")


@app.route("/anime_list")
def anime_list():
    selected_anime = request.args.get("search", "")

    if selected_anime:
        anime_suggested = recommendations(selected_anime)
    print(anime_suggested)

    return render_template("second_page.html")



if __name__ == "__main__":
    app.run(debug=True)

