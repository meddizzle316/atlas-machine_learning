-- a script that displays the max temperature of each state (ordered by State name).

SELECT state, MAX(value) as max_temp

FROM hbtn_0c_0.temperatures

GROUP BY state

ORDER BY state;