-- a SQL script that creates a table users following these requirements:

CREATE TABLE IF NOT EXISTS users (
    id INT NOT NULL GENERATED ALWAYS AS IDENTITY,
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255),
    PRIMARY KEY (id)
);