-- a script that lists all shows from hbtn_0d_tvshows_rate by their rating.

SELECT
    tv_shows.title,
    COALESCE(SUM(ratings.rating), 0) AS rating_sum
FROM tv_shows
LEFT JOIN ratings ON tv_shows.id = ratings.tv_show_id
GROUP BY tv_shows.id
ORDER BY rating_sum DESC;