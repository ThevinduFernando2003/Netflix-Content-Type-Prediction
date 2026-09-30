"""Hand-built catalogue used by the unit tests.

The rows are written by this module so the committed CSV stays reviewable and
includes every defect the cleaning tests assert.
"""

from __future__ import annotations

import csv
from pathlib import Path


def rows() -> list[dict[str, str]]:
    """Return 80 catalogue rows, about 70 percent movies."""
    data: list[dict[str, str]] = [
        {
            "show_id": "m1",
            "type": "Movie",
            "title": "Shifted Runtime",
            "director": "Ava Patel",
            "cast": "Ava Stone, Liam Ortiz",
            "country": "United States",
            "date_added": "September 25, 2021",
            "release_year": "2019",
            "rating": "74 min",
            "duration": "",
            "listed_in": "Dramas",
            "description": "A heist pilot flies one last cargo over the Andes.",
        },
        {
            "show_id": "m2",
            "type": "Movie",
            "title": "No Director Listed",
            "director": "",
            "cast": "Mia Chen",
            "country": "United States",
            "date_added": "9/1/2020",
            "release_year": "2018",
            "rating": "PG-13",
            "duration": "102 min",
            "listed_in": "Comedies",
            "description": "A heist goes wrong in a quiet coastal town.",
        },
        {
            "show_id": "m3",
            "type": "Movie",
            "title": "Empty Cast List",
            "director": "Noah Kim",
            "cast": "",
            "country": "United States",
            "date_added": "9/2/2020",
            "release_year": "2017",
            "rating": "PG",
            "duration": "95 min",
            "listed_in": "Children & Family Movies",
            "description": "A heist of lost toys brings a family back together.",
        },
        {
            "show_id": "m4",
            "type": "Movie",
            "title": "No Country Listed",
            "director": "Noah Kim",
            "cast": "Mia Chen",
            "country": "",
            "date_added": "9/3/2020",
            "release_year": "2016",
            "rating": "R",
            "duration": "110 min",
            "listed_in": "Thrillers",
            "description": "A heist unravels inside an unnamed capital.",
        },
        {
            "show_id": "m5",
            "type": "Movie",
            "title": "Missing Rating",
            "director": "Noah Kim",
            "cast": "Mia Chen",
            "country": "United States",
            "date_added": "9/4/2020",
            "release_year": "2015",
            "rating": "",
            "duration": "88 min",
            "listed_in": "Dramas",
            "description": "A heist leaves one witness and no official rating.",
        },
        {
            "show_id": "m6",
            "type": "Movie",
            "title": "Standard Feature",
            "director": "Noah Kim",
            "cast": "Mia Chen, Ava Stone",
            "country": "United States",
            "date_added": "9/5/2020",
            "release_year": "2014",
            "rating": "PG-13",
            "duration": "90 min",
            "listed_in": "Action & Adventure",
            "description": "A heist crew races a storm back to the harbor.",
        },
        {
            "show_id": "m7",
            "type": "Movie",
            "title": "Two Countries",
            "director": "Noah Kim",
            "cast": "Mia Chen",
            "country": "United States, Canada",
            "date_added": "9/6/2020",
            "release_year": "2013",
            "rating": "PG-13",
            "duration": "101 min",
            "listed_in": "Dramas, International Movies",
            "description": "A heist crosses the border between two cities.",
        },
        {
            "show_id": "m8",
            "type": "Movie",
            "title": "Five Genres",
            "director": "Noah Kim",
            "cast": "Mia Chen, Ava Stone, Liam Ortiz",
            "country": "United States",
            "date_added": "9/7/2020",
            "release_year": "2012",
            "rating": "PG-13",
            "duration": "120 min",
            "listed_in": "Dramas, Comedies, Independent Movies, Thrillers, Sports Movies",
            "description": "A heist, a joke, a match, and a chase share one night.",
        },
        {
            "show_id": "m9",
            "type": "Movie",
            "title": "Blank Synopsis",
            "director": "Noah Kim",
            "cast": "Mia Chen",
            "country": "United States",
            "date_added": "9/8/2020",
            "release_year": "2011",
            "rating": "PG",
            "duration": "80 min",
            "listed_in": "Documentaries",
            "description": " ",
        },
        {
            "show_id": "m10",
            "type": "Movie",
            "title": "Amélie",
            "director": "Jean-Pierre Jeunet",
            "cast": "Audrey Tautou",
            "country": "France",
            "date_added": "9/9/2020",
            "release_year": "2001",
            "rating": "R",
            "duration": "122 min",
            "listed_in": "Comedies, International Movies, Romantic Movies",
            "description": "A heist of small kindnesses changes a Paris neighborhood.",
        },
        {
            "show_id": "m11",
            "type": "Movie",
            "title": "Hello, Goodbye",
            "director": "Noah Kim",
            "cast": "Mia Chen",
            "country": "United States",
            "date_added": "9/10/2020",
            "release_year": "2010",
            "rating": "PG-13",
            "duration": "99 min",
            "listed_in": "Dramas, Romantic Movies",
            "description": "A heist of letters reunites two people who left the harbor.",
        },
        {
            "show_id": "m12",
            "type": "Movie",
            "title": "The Heist",
            "director": "Noah Kim",
            "cast": "Mia Chen, Ava Stone",
            "country": "United States",
            "date_added": "9/11/2020",
            "release_year": "2009",
            "rating": "PG-13",
            "duration": "108 min",
            "listed_in": "Action & Adventure, Comedies",
            "description": "A heist film about one night in a museum.",
        },
        {
            "show_id": "t1",
            "type": "TV Show",
            "title": "The Heist",
            "director": "",
            "cast": "Mia Chen, Ava Stone, Liam Ortiz, Noah Blake",
            "country": "United Kingdom",
            "date_added": "9/12/2020",
            "release_year": "2020",
            "rating": "TV-MA",
            "duration": "2 Seasons",
            "listed_in": "Crime TV Shows, International TV Shows, TV Dramas",
            "description": "In this harbor series a crew returns every season for another job.",
        },
        {
            "show_id": "t2",
            "type": "TV Show",
            "title": "Harbor Nights",
            "director": "Not Given",
            "cast": "Liam Ortiz",
            "country": "United Kingdom",
            "date_added": "9/13/2020",
            "release_year": "2019",
            "rating": "TV-14",
            "duration": "1 Season",
            "listed_in": "TV Dramas, TV Mysteries",
            "description": "A harbor detective works the docks across one season.",
        },
    ]

    for index in range(13, 57):
        data.append(
            {
                "show_id": f"m{index}",
                "type": "Movie",
                "title": f"Harbor Heist {index}",
                "director": "Noah Kim",
                "cast": "Mia Chen, Ava Stone",
                "country": "United States",
                "date_added": "1/5/2021",
                "release_year": str(1990 + (index % 25)),
                "rating": "PG-13",
                "duration": f"{80 + index} min",
                "listed_in": "Dramas, Comedies",
                "description": f"A heist story number {index} about a pilot and a harbor town.",
            }
        )
    for index in range(3, 25):
        data.append(
            {
                "show_id": f"t{index}",
                "type": "TV Show",
                "title": f"Harbor Lane {index}",
                "director": "",
                "cast": "Liam Ortiz, Mia Chen, Ava Stone",
                "country": "United Kingdom" if index % 2 == 0 else "United States",
                "date_added": "March 2, 2021",
                "release_year": str(2010 + (index % 10)),
                "rating": "TV-MA",
                "duration": "1 Season",
                "listed_in": "TV Dramas, International TV Shows",
                "description": f"A harbor series number {index} follows the same street each season.",
            }
        )
    data.append({**data[0], "title": "Duplicate Should Drop"})
    return data


def main() -> None:
    destination = Path(__file__).with_name("mini_netflix.csv")
    fieldnames = [
        "show_id",
        "type",
        "title",
        "director",
        "cast",
        "country",
        "date_added",
        "release_year",
        "rating",
        "duration",
        "listed_in",
        "description",
    ]
    table = rows()
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)
    movies = sum(row["type"] == "Movie" for row in table)
    print(f"Wrote {len(table)} rows ({movies} movies) to {destination}")


if __name__ == "__main__":
    main()
