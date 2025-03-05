-- Write a script that lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each.

SELECT
    genres.name AS genre,
    COUNT(tv_shows.id) AS number_of_shows
FROM genres
JOIN tv_shows ON tv_shows.genre_id = genres.id
GROUP BY genres.id
ORDER BY number_of_shows DESC;